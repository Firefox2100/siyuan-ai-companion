# SiYuan AI Companion

A companion service to use SiYuan note as a knowledge base with OpenAI APIs.

## License and disclaimer

This project is licensed under the GNU General Public License v3.0 (GPL-3.0). See the [licence file](LICENSE) for details.

This project is not affiliated with or endorsed by SiYuan or OpenAI. It uses their API format, but is not guaranteed to be compatible with their services. If the API format is changed, this project may not work as expected. Even if it works, there is still a possibility that it may adversely affect your data, or your OpenAI service account. Use at your own risk, and be sure to understand what you are doing before using this project.

You are welcomed to inspect the code, propose changes, report bugs, or engage or support the project in any way.

## Motivation and design considerations

This project is designed to solve one of my problems: I switched to SiYuan from Obsidian, and I wanted to use my notes as a knowledge vault for my local running LLMs. For Obsidian, this is as simple as embedding the files and use them for RAG; however SiYuan stores its data in a JSON based database, and the raw files contain a lot of metadata that is not useful for LLMs. This project is designed as a proxy to OpenAI compatible APIs, by providing a layer to perform the RAG and add the note contents to the prompt before sending it to the LLM service. It needs to:

- Be minimally intrusive on SiYuan: SiYuan imposes a file-renaming based conflict detection feature. It repeatedly rename a file and check for duplications to ensure that it has exclusive access to its workspace. This companion app needs to work around that restriction to prevent SiYuan from exiting.
- Able to self-host: The stack should be simple enough and uses minimal resources to allow self-hosting on a mid to low-end server.
- Handle as much as data format and structure in SiYuan as possible: SiYuan is not a Markdown based note-taking app, and it allows for more complicated data types and relationships between them. This project should be able to handle as much of that as possible, and provide a way to extract the data in a format that is useful for LLMs.

Based on the considerations, it decided to:

- Instead of reading the raw files of SiYuan and risk corrupting the database, it will use the SiYuan API to read the data. SiYuan exposes a series of REST API, including one SQL-like query endpoint. This project uses the SQL endpoint to read the data block-wise, without ever touching the raw files. This however **requires a SiYuan instance to be running when using the companion app**, for this I recommend deploying a docker container with the SiYuan image.
- Use Qdrant as the vector database to store the embeddings. Qdrant is a high-performance vector database that is easy to set up and use. It provides an official docker image, and can be used in-memory or with persistent storage, depending on the scale and development need.
- Use Quart and APScheduler to provide an ASGI server and a background non-blocking scheduler to handle the API requests and the background tasks.

## Features

- [x] Support for OpenAI API proxy
- [x] Support for getting notes from SiYuan
- [x] Support for querying note blocks via last updated time
- [x] Support for embedding note blocks and store them in Qdrant
- [x] Support for querying Qdrant for similar blocks, reconstruct the original note by querying SiYuan API, and generate prompt for OpenAI API

## Installation

This project can be used as a docker image, or run from source code. To run it from source code, install dependencies with pip (preferably in a virtual environment):

```bash
pip install .
```

If used in production, it's recommended to use a production-ready ASGI server, and a standalone Qdrant instance. Assuming the Qdrant is running on `localhost:6333`, and the SiYuan instance is running on `localhost:6806`, you can run the server with:

```bash
export SIYUAN_URL="http://localhost:6806"
export SIYUAN_TOKEN="your-siyuan-token"
export QDRANT_LOCATION="localhost:6333"
export QDRANT_COLLECTION_NAME="siyuan_ai_companion"
export OPENAI_URL="https://api.openai.com/v1/"

pip install hypercorn
hypercorn siyuan_ai_companion.asgi:application
```

If using docker, the image has been uploaded to Docker Hub, and can be used with:

```yaml
services:
  siyuan-ai-companion:
    image: firefox2100/siyuan-ai-companion:latest
    restart: always
    container_name: siyuan-ai-companion
    environment:
      - SIYUAN_URL=http://siyuan:6806
      - SIYUAN_TOKEN=your-siyuan-token
      - QDRANT_LOCATION=qdrant:6333
      - QDRANT_COLLECTION_NAME=siyuan_ai_companion
      - OPENAI_URL=https://api.openai.com/v1/
    ports:
      - "8000:8000"

  qdrant:
    image: qdrant/qdrant:latest
    restart: always
    container_name: qdrant
    configs:
      - source: qdrant_config
        target: /qdrant/config/production.yaml
    volumes:
      - ./qdrant_data:/qdrant/storage
  
  siyuan:
    image: b3log/siyuan:latest
    command: ['--workspace=/siyuan/workspace/', '--accessAuthCode=your-auth-code']
    ports:
      - 6806:6806
    volumes:
      - /siyuan/workspace:/siyuan/workspace
    restart: unless-stopped
    environment:
      # A list of time zone identifiers can be found at https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
      - TZ=${YOUR_TIME_ZONE}
      - PUID=${YOUR_USER_PUID}  # Customize user ID
      - PGID=${YOUR_USER_PGID}  # Customize group ID

configs:
  qdrant_config:
    content: |
      log_level: INFO
```

## Configuration and usage

The application uses environment variables to configure the connection to SiYuan and Qdrant. The following environment variables are used:

- SIYUAN_URL: The URL of the SiYuan instance. This is required to read the data from SiYuan. It should be a URL with protocol, e.g. `http://localhost:6806`.
- SIYUAN_TOKEN: The token to access the SiYuan API. This is required to read the data from SiYuan. This is NOT the docker auth code, but the one you see within the setting page.
- QDRANT_LOCATION: The URL of the Qdrant instance. This is required to store the embeddings. If using in-memory Qdrant, this can be set to `:memory:`.
- QDRANT_COLLECTION_NAME: The name of the collection to use in Qdrant. This is required to store the embeddings. If the collection does not exist, it will be created automatically.
- OPENAI_URL: The URL of the OpenAI compatible API. This does not need to be reachable from the outside. So if you host your own LLM service, you can set this to a local address, or even a docker network address. As long as it's reachable from the container.
- FORCE_UPDATE_INDEX: Set this to `true` to force the companion to rebuild the index everytime it restarts. This is useful for development or recovering from a corrupted index.

All requests sent to the API are passed to the OpenAI compatible API, with all original headers. This service does not check whether your request format is correct, have the right headers, or anything related to the API, except for the prompt field.

This service uses only `SELECT` queries and only the SQL endpoint. In theory, it should not affect the note data or damage it in any way. However, it is recommended to backup your data before using this service, just in case. If unforeseen errors occur, it may send more than one request to the OpenAI API, which may incur additional costs; if possible, use a separate API key when using this service, one that is rate limited or have a low spending cost.
