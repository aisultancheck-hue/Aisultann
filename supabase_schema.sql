create extension if not exists vector;

create table if not exists documents (
  id text primary key,
  source text not null,
  chunk_index integer not null,
  content text not null,
  embedding vector(384) not null,
  created_at timestamptz default now()
);

create index if not exists documents_embedding_idx
on documents using ivfflat (embedding vector_cosine_ops)
with (lists = 100);

create or replace function match_documents(
  query_embedding vector(384),
  match_count int default 5
)
returns table (
  id text,
  source text,
  chunk_index integer,
  content text,
  similarity float
)
language sql stable
as $$
  select
    documents.id,
    documents.source,
    documents.chunk_index,
    documents.content,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  order by documents.embedding <=> query_embedding
  limit match_count;
$$;
