-- ============================================================================
-- Migration: long-term dataset registry table (run once in Supabase SQL editor)
-- Safe to re-run — everything is idempotent.
-- ============================================================================

create table if not exists public.registered_datasets (
    name        text primary key,
    hf_id       text not null,
    kind        text,
    rows        integer default 0,
    updated_at  timestamptz,
    synced_at   timestamptz not null default now()
);

alter table public.registered_datasets enable row level security;

drop policy if exists "neurochat_anon_registered_datasets" on public.registered_datasets;
create policy "neurochat_anon_registered_datasets"
    on public.registered_datasets for all to anon using (true) with check (true);
