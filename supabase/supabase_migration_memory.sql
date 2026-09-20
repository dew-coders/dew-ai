-- ============================================================================
-- Migration: long-term user memory table (run once in Supabase SQL editor)
-- Safe to re-run — everything is idempotent.
-- ============================================================================

create table if not exists public.user_memory (
    id          uuid primary key,
    user_id     uuid not null references public.users(id) on delete cascade,
    kind        text not null,               -- name | location | likes | note …
    value       text not null,
    source      text default 'user',
    created_at  timestamptz not null default now()
);

create index if not exists idx_cloud_memory_user on public.user_memory(user_id);

alter table public.user_memory enable row level security;

drop policy if exists "neurochat_anon_user_memory" on public.user_memory;
create policy "neurochat_anon_user_memory"
    on public.user_memory for all to anon using (true) with check (true);
