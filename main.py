import uvicorn
from os import makedirs, path
from socket import AF_INET, AF_INET6, SOCK_STREAM, socket

from api import app
from typing import Optional


def main(
    host: Optional[str] = None,
    port: int = 0,
    ipv6: bool = False,
    portfile: Optional[str] = None,
    access_log: bool = False,
):
    server = uvicorn.Server(
        config=uvicorn.Config(app, reload=True, access_log=access_log)
    )

    s = socket(AF_INET6 if ipv6 else AF_INET, SOCK_STREAM)
    s.bind((host or ("::1" if ipv6 else "127.0.0.1"), port))
    host = s.getsockname()[0]
    port = s.getsockname()[1]

    print("Listening on %s:%s" % (("[%s]" % host) if ipv6 else host, port))

    if portfile is not None:
        makedirs(path.dirname(path.realpath(portfile)), exist_ok=True)
        with open(portfile, "w") as file:
            file.write("%d" % port)

    server.run([s])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=0)
    parser.add_argument("--host", type=str)
    parser.add_argument("--portfile", type=str)
    parser.add_argument("--ipv6", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--access-log", action=argparse.BooleanOptionalAction, default=False
    )
    args = vars(parser.parse_args())
    main(**args)
