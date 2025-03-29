# 思源AI助手

一个用于将 SiYuan 笔记作为知识库，与 OpenAI API 配合使用的辅助服务。

## 许可证与免责声明

本项目遵循 GNU 通用公共许可证 v3.0（GPL-3.0）。详情请参阅 许可证文件。

本项目与 SiYuan 或 OpenAI 无任何隶属或官方合作关系。尽管使用了它们的 API 格式，但并不保证与其服务完全兼容。如果 API 格式发生变更，本项目可能无法按预期运行。即使运行正常，也存在可能对您的数据或 OpenAI 服务账户造成不利影响的风险。请在了解项目工作原理的前提下使用，风险自负。

欢迎您查看代码、提出修改建议、报告错误，或以任何方式参与或支持本项目。

## 开发动机与设计考虑

本项目旨在解决我个人的需求：我从 Obsidian 转向了思源笔记，希望将我的笔记用作本地运行的大语言模型的知识库。在 Obsidian 中，本地的LLM服务可以直接读取文件，然后用于RAG（Retrival Augmented Generation)；但思源的笔记存储在基于 JSON 的数据库中，原始文件中包含大量对大模型无用的元数据。因此我设计了本项目，它作为一个 OpenAI 兼容 API 的代理层，负责执行 RAG，将笔记内容添加到提示词中，然后发送给大语言模型服务。它需要：

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
- [x] 支持嵌入笔记块并存储于 Qdrant
- [x] 支持从 Qdrant 查询相似块、通过思源 API 重构原始笔记并生成 OpenAI API 所需提示词

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

应用通过环境变量配置连接信息。所使用的环境变量如下：

- `SIYUAN_URL`: SiYuan 实例的 URL，用于读取数据，需包含协议（如 http://localhost:6806）。
- `SIYUAN_TOKEN`: 访问 SiYuan API 所需的令牌。注意，这不是 Docker 启动时的授权码，而是在设置页面中看到的 API Token。
- `QDRANT_LOCATION`: Qdrant 实例地址。用于存储嵌入向量。如使用内存模式，可设置为 :memory:。
- `QDRANT_COLLECTION_NAME`: Qdrant 中使用的集合名称。若集合不存在，将自动创建。
- `OPENAI_URL`: OpenAI 兼容 API 的地址。不需要外部可访问，可设置为本地地址或容器网络地址，只需服务容器可访问即可。
- `FORCE_UPDATE_INDEX`: 若设为 true，则每次重启时都会强制重建索引。适用于开发或修复损坏的索引。

所有发送至该服务的 API 请求将原样转发至 OpenAI 兼容 API，包括所有原始请求头。本服务不验证请求格式或请求头，仅处理 prompt 字段。

本服务仅使用 SELECT 语句，并仅调用 SQL 接口。理论上不会修改或破坏任何笔记数据。但为安全起见，建议在使用前备份数据。若出现错误，可能导致向 OpenAI API 发送多次请求，从而产生额外费用；建议使用单独的、费用受限的 API 密钥运行该服务。
