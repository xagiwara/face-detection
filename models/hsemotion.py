from os import listdir, path
from os.path import isfile
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
from env import DATA_HSEMOTION


class HSEmotionResult:
    emotion: str
    scores: dict[str, float]

    def __init__(self, emotion: str, scores: dict[str, float]) -> None:
        self.emotion = emotion
        self.scores = scores


# Original is hsemotion/hsemotion/facial_emotions.py
class HSEmotion:
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.is_mtl = "_mtl" in model_name
        if "_7" in model_name:
            self.idx_to_class = {
                0: "Anger",
                1: "Disgust",
                2: "Fear",
                3: "Happiness",
                4: "Neutral",
                5: "Sadness",
                6: "Surprise",
            }
        else:
            self.idx_to_class = {
                0: "Anger",
                1: "Contempt",
                2: "Disgust",
                3: "Fear",
                4: "Happiness",
                5: "Neutral",
                6: "Sadness",
                7: "Surprise",
            }

        self.img_size = 224 if "_b0_" in model_name else 260
        self.test_transforms = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        model_path = path.join(DATA_HSEMOTION, model_name)

        if device == "cpu":
            model = torch.load(model_path, map_location=torch.device("cpu"))
        else:
            model = torch.load(model_path)

        if isinstance(model.classifier, torch.nn.Sequential):
            self.classifier_weights = model.classifier[0].weight.cpu().data.numpy()
            self.classifier_bias = model.classifier[0].bias.cpu().data.numpy()
        else:
            self.classifier_weights = model.classifier.weight.cpu().data.numpy()
            self.classifier_bias = model.classifier.bias.cpu().data.numpy()

        model.classifier = torch.nn.Identity()
        model = model.to(device)
        self.model = model.eval()

    def get_probab(self, features):
        x = (
            np.dot(features, np.transpose(self.classifier_weights))
            + self.classifier_bias
        )
        return x

    def extract_features(self, face_img):
        img_tensor = self.test_transforms(Image.fromarray(face_img))
        img_tensor.unsqueeze_(0)
        features = self.model(img_tensor.to(self.device))
        features = features.data.cpu().numpy()
        return features

    def predict_emotions(self, face_img, logits=True):
        features = self.extract_features(face_img)
        scores = self.get_probab(features)[0]
        if self.is_mtl:
            x = scores[:-2]
        else:
            x = scores
        pred = np.argmax(x)

        if not logits:
            e_x = np.exp(x - np.max(x)[np.newaxis])
            e_x = e_x / e_x.sum()[None]
            if self.is_mtl:
                scores[:-2] = e_x
            else:
                scores = e_x
        return self.idx_to_class[pred], scores

    def extract_multi_features(self, face_img_list):
        imgs = [
            self.test_transforms(Image.fromarray(face_img))
            for face_img in face_img_list
        ]
        features = self.model(torch.stack(imgs, dim=0).to(self.device))
        features = features.data.cpu().numpy()
        return features

    def predict_multi_emotions(self, face_img_list, logits=True):
        features = self.extract_multi_features(face_img_list)
        scores = self.get_probab(features)
        if self.is_mtl:
            preds = np.argmax(scores[:, :-2], axis=1)
        else:
            preds = np.argmax(scores, axis=1)
        if self.is_mtl:
            x = scores[:, :-2]
        else:
            x = scores

        if not logits:
            e_x = np.exp(x - np.max(x, axis=1)[:, np.newaxis])
            e_x = e_x / e_x.sum(axis=1)[:, None]
            if self.is_mtl:
                scores[:, :-2] = e_x
            else:
                scores = e_x

        return [self.idx_to_class[pred] for pred in preds], scores


model_cached: dict[str, HSEmotion] = {}


def load_model(model_name: str, device: str):
    key = "%s/%s" % (model_name, device)

    if key in model_cached:
        return model_cached[key]

    model = HSEmotion(model_name, device)

    print("hsemotion: %s model loaded with %s." % (model_name, device))
    model_cached[key] = model
    return model


def models():
    return [f for f in listdir(DATA_HSEMOTION) if isfile(path.join(DATA_HSEMOTION, f))]


def hsemotion(imgs: list[cv2.typing.MatLike], model: str, cuda: str):
    model = load_model(model, cuda)
    emotion, scores = model.predict_multi_emotions(imgs)

    return [
        HSEmotionResult(
            emotion[j],
            {
                model.idx_to_class[i]: scores[j][i].item()
                for i in range(len(model.idx_to_class))
            },
        )
        for j in range(len(imgs))
    ]
