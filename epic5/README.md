# Epic 5 — Database Layer

`ViolationDBWriter` reads MySQL settings from `app/.env`, upserts events by
their unique `event_id`, and copies full-frame and plate evidence into
Laravel's `storage/app/public/violations` directory.

Failed database writes are retried and then queued in
`epic5/pending_db_events.json`. A later run flushes queued records.

Connection and historical JSON import:

```powershell
python epic5\db_writer.py --ping
python epic5\db_writer.py --import-json violation_events.json
```

Create Laravel's public storage link once:

```powershell
cd app
php artisan storage:link
```
