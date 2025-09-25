# Container Runner

import docker

class ContainerRunner:
    def run(self, image: str, command: str) -> str:
        client = docker.from_env()
        container = client.containers.run(image, command, detach=True)
        result = container.wait()
        return result["StatusCode"]