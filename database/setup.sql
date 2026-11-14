CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS cadvpdm_knowledge (
    id BIGSERIAL PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL DEFAULT 0,
    title TEXT,
    content TEXT NOT NULL,
    site TEXT,
    application TEXT,
    error_code TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding VECTOR(384),
    UNIQUE(source_type, source_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_cadvpdm_source_type ON cadvpdm_knowledge(source_type);
CREATE INDEX IF NOT EXISTS idx_cadvpdm_application ON cadvpdm_knowledge(application);
CREATE INDEX IF NOT EXISTS idx_cadvpdm_site ON cadvpdm_knowledge(site);
CREATE INDEX IF NOT EXISTS idx_cadvpdm_error_code ON cadvpdm_knowledge(error_code);
