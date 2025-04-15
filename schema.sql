CREATE TABLE blocks (
    id           VARCHAR(32)  PRIMARY KEY,                -- Content block ID
    parent_id    VARCHAR(32),                             -- Parent block ID (nullable if document block)
    root_id      VARCHAR(32)  NOT NULL,                   -- Root block ID (document block ID)
    hash         VARCHAR(64),                             -- Summary checksum of the content
    box          VARCHAR(32)  NOT NULL,                   -- Notebook ID
    path         TEXT        NOT NULL,                    -- Document path where the content block is located
    hpath        TEXT,                                    -- Human-readable document path
    name         TEXT,                                    -- Content block name
    alias        TEXT,                                    -- Content block alias
    memo         TEXT,                                    -- Content block memo
    content      TEXT,                                    -- Text with Markdown markers removed
    fcontent     TEXT,                                    -- First child block text with Markdown markers removed
    markdown     TEXT,                                    -- Full Markdown content
    length       INTEGER,                                 -- Length of `fcontent`
    type         VARCHAR(16),                             -- Content block type (refers to external enum or table)
    subtype      VARCHAR(16),                             -- Content block subtype (refers to external enum or table)
    ial          TEXT,                                    -- Inline attribute list
    sort         INTEGER DEFAULT 0,                       -- Sort value, smaller = higher priority
    created      CHAR(14)  NOT NULL,                      -- Creation timestamp in `yyyyMMddHHmmss` format
    updated      CHAR(14)  NOT NULL,                      -- Update timestamp in `yyyyMMddHHmmss` format
);
