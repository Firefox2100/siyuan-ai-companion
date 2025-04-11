# 思源AI助手

[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=Firefox2100_siyuan-ai-companion&metric=bugs)](https://sonarcloud.io/summary/new_code?id=Firefox2100_siyuan-ai-companion) [![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=Firefox2100_siyuan-ai-companion&metric=reliability_rating)](https://sonarcloud.io/summary/new_code?id=Firefox2100_siyuan-ai-companion) [![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=Firefox2100_siyuan-ai-companion&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=Firefox2100_siyuan-ai-companion) [![Coverage](https://sonarcloud.io/api/project_badges/measure?project=Firefox2100_siyuan-ai-companion&metric=coverage)](https://sonarcloud.io/summary/new_code?id=Firefox2100_siyuan-ai-companion)

[English](README.md)

一个用于将 SiYuan 笔记作为知识库，与 OpenAI API 配合使用的辅助服务。

## 许可证与免责声明

本项目遵循 GNU 通用公共许可证 v3.0（GPL-3.0）。详情请参阅 许可证文件。

本项目与 SiYuan 或 OpenAI 无任何隶属或官方合作关系。尽管使用了它们的 API 格式，但并不保证与其服务完全兼容。如果 API 格式发生变更，本项目可能无法按照预期的方式运行。即使运行正常，也存在可能对您的数据或 OpenAI 服务账户造成不利影响的风险。请在了解项目工作原理的前提下使用，风险自负。

欢迎您查看代码、提出修改建议、报告错误，或以任何方式参与或支持本项目。

## 开发动机与设计考量

本项目旨在解决我个人的需求：我从 Obsidian 转向了思源笔记，希望将我的笔记用作本地运行的大语言模型的知识库。在 Obsidian 中，本地的LLM服务可以直接读取文件，然后用于RAG（Retrival Augmented Generation)；但思源的笔记存储在基于 JSON 的数据库中，原始文件中包含大量对LLM无用，反而会产生干扰的元数据。因此我开发了本项目，它作为一个 OpenAI 兼容 API 的代理层，负责执行 RAG，将笔记内容添加到提示词中，然后发送给大语言模型服务。它应该：

- 尽量少地干扰思源笔记：思源笔记采用基于文件重命名的冲突检测机制。它会反复重命名文件并检测是否有冲突和重复文件，以确保对工作区的独占访问。本应用需要绕过这一限制，以防止思源异常退出。
- 支持自部署：技术栈应尽可能简单且资源占用低，以便在中低端服务器上运行。
- 尽可能多地处理思源的数据结构：思源并非基于 Markdown 的笔记应用，它支持更复杂的数据类型及其关系。本项目应尽可能支持这些结构，并以适合 LLM 的格式提取数据。

基于以上考虑，项目采取了以下设计：

- 使用思源 API 读取数据，而非直接读取原始文件，以避免破坏数据库。思源提供了 REST API，其中包含一个类似于 SQL 的查询接口。本项目使用该 SQL 接口按块读取数据，从不直接操作原始文件。但这表示在使用本应用时需要一个正在运行的思源实例，建议通过 Docker 容器部署思源镜像。
- 使用 Qdrant 作为向量数据库存储嵌入。Qdrant 是一个高性能的向量数据库，易于部署和使用，提供官方 Docker 镜像，可以选择在内存中运行还是将数据保存至文件系统。
- 使用 Quart 和 APScheduler 构建 ASGI 服务，并处理 API 请求及后台任务。

## 功能特性

- [x] 支持 OpenAI API 代理
- [x] 支持从思源获取笔记内容
- [x] 支持按更新时间查询笔记块
- [x] 支持嵌入笔记块并存储于 Qdrant 中
- [x] 支持从 Qdrant 查询相似块、通过思源 API 重构原始笔记并生成 OpenAI API 所需提示词
- [x] 支持列出所有的资源文件（嵌入笔记的媒体和数据附件）
- [x] 支持直接对附件内的音频进行语音转写，和说话人识别。这一功能本不属于代理类服务，但是因为目前没有一个专门的用于语音识别的自部署服务，我就自己写了一个。

## 安装

本项目可作为 Docker 镜像使用，也可以从源码运行。若从源码运行，建议在虚拟环境中通过 pip 安装依赖：

```bash
pip install .
```

用于生产环境时，推荐使用生产级别的 ASGI 服务器及独立的 Qdrant 实例。假设 Qdrant 运行在 `localhost:6333`，思源运行在 `localhost:6806`，可以如下启动服务：

```bash
export SIYUAN_URL="http://localhost:6806"
export SIYUAN_TOKEN="your-siyuan-token"
export QDRANT_LOCATION="localhost:6333"
export QDRANT_COLLECTION_NAME="siyuan_ai_companion"
export OPENAI_URL="https://api.openai.com/v1/"

pip install hypercorn
hypercorn siyuan_ai_companion.asgi:application
```

如果使用 Docker，镜像已上传至 Docker Hub，可通过以下方式使用：

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

## 配置与使用

这一服务通过环境变量配置连接信息。所使用的环境变量如下：

- `SIYUAN_URL`: 思源实例的 URL，用于读取数据，需包含协议（如 http://localhost:6806）。
- `SIYUAN_TOKEN`: 访问思源 API 所需的令牌。注意，这不是 Docker 网页登录的 Token，而是在设置页面中看到的 API Token。
- `QDRANT_LOCATION`: Qdrant 实例地址。用于存储嵌入向量。如使用内存模式，可设置为 :memory:。
- `QDRANT_COLLECTION_NAME`: Qdrant 中使用的集合名称。若集合不存在，将自动创建。
- `OPENAI_URL`: OpenAI 兼容 API 的地址。不需要外部可访问，可设置为本地地址或容器网络地址，只需服务容器可访问即可。
- `OPENAI_TOKEN`: OpenAI API 的访问令牌。用于访问 OpenAI API。如果不设置，那么不会发送 `Authorization` 标头。一般自己部署的LLM服务都没有这个配置，但是如果反向代理配置了这个标头，或者使用 OpenAI 官方的接口，那么这个配置是必须的。
- `COMPANION_TOKEN`: 这个服务自己的访问令牌。用于访问本服务的 API。可以设置为任意长度的任意值，只要HTTP请求能够发送这个标头即可。因为本服务能够读取和创建你的笔记数据，如果这个服务能够从互联网访问，建议配置这个令牌，否则任何人都有可能获取或者修改你的笔记数据。
- `WHISPER_WORKERS`: 语音转写的工作线程数（`faster-whisper` 的配置）。默认是1。这些线程会隶属于单独的一个子进程，所以与主服务的线程互相独立。配置的时候建议考虑最大核心数，和一些CPU的超线程功能。同属需要注意的是，如果没有GPU支持（上传的 Docker 镜像完全没有CUDA支持），`pyannote` 也会用CPU进行识别，需要预留核心数。
- `HUGGINGFACE_HUB_TOKEN`: 用于访问 Hugging Face Hub 的令牌。用于下载语音转写模型。需要注意的是，用到的模型有自己的用户协议，如果没有在网页上选择接受协议，那么在下载模型时会失败。
- `SIYUAN_TRANSCRIBE_NOTEBOOK`: 默认用于保存语音转写结果的笔记本名称。如果留空，那么每次转写请求必须包含指定的笔记本。
- `FORCE_UPDATE_INDEX`: 若设为 true，则每次重启时都会强制重建索引。适用于开发或修复损坏的索引。

所有发送至该服务的 API 请求将原样转发至 OpenAI 兼容 API，包括所有原始请求头。本服务不验证请求格式或请求头，仅处理 prompt 字段。

本服务仅使用 SELECT 语句来调用 SQL 接口。理论上不会修改或破坏任何笔记数据，只会读取笔记，下载音频附件，和创建新笔记（不会覆盖）。但为安全起见，建议在使用前备份数据。同时，若出现错误，可能导致向 OpenAI API 发送多次请求，从而产生额外费用；建议使用单独的、费用受限的 API 密钥运行该服务。

## 怎么使用本服务的功能

### 请求鉴权

如果配置了 `COMPANION_TOKEN`，那么所有请求都需要在请求头中包含 `Authorization` 字段，格式为 `Bearer {COMPANION_TOKEN}`。比如：

```bash
curl -H "Authorization: Bearer your-token-here" https://your.companion.com/assets/
```

如果上游的 OpenAI 接口也需要一个令牌，可以配置 `OPENAI_TOKEN`，这个令牌会被替换掉。如果没有配置，这个令牌会被删除。无论如果，这个令牌都不会被转发给别的服务，所以无需担心泄露。

### 检索增强生成（RAG）

本服务的核心功能就是用思源笔记实现 RAG。这一功能不需要用户的任何额外操作，本服务会自动完成，即：

本服务每5分钟会自动查询思源数据库，看是否有内容块被更新过。若有更新，则会将这些内容块嵌入到 Qdrant 中。这是一个后台功能，同时也只会为更新了的块生成新的嵌入向量。这是为了确保不浪费服务器资源。

在收到生成请求时，本服务会自动从 Qdrant 中查询与请求内容最相似的块，并且从思源获取**整个笔记文档**然后发送给上游 OpenAI 服务。

> **注意**：思源笔记采用“块”的概念来存储笔记内容。笔记块是笔记的最小单位，例如一个标题，一个自然段，列表里的一个列表项，等等。在进行更新查询的时候获取的结果也是以块为单位的。为了确保每次更新内容最少，本服务在嵌入和索引的时候也是以块为单位，所以无法在嵌入之前进行语义分段。结果就是，整篇笔记都需要发送给上游的 OpenAI 服务。我有计划再用一个数据库保存按语义分段的信息，这样就能提取一小段发给上游服务。如果你愿意帮助我实现这个功能，欢迎提交 PR。

### 语音转写

这个功能主要是给我自己使用的，是为了将我的会议录音和音频笔记转成文字。向专门的 API 发送请求可以试本服务下载一个指定的音频（用完之后会自动删掉），通过 `faster-whisper` 和 `pyannote` 进行转写和说话人识别。转写完成后，会将结果保存到思源笔记中。

> **注意**：我也知道每次通过命令行或者 Postman 来发送请求很麻烦，所以计划做一个 UI 界面来处理语音转写和 LLM 聊天功能，或者是直接集成到思源笔记插件里。但是我几乎不会 Vue.js，也完全不会 TypeScript，所以如果你有开发思源笔记插件的经验的话，欢迎提交 PR。
