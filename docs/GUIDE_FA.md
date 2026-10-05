# ASTRA TRADER — راهنمای ساده MT4

این نسخه **SIMULATION / READ-ONLY** است: MT4 فقط منبع داده است و ASTRA هیچ سفارش واقعی به MT4 نمی‌فرستد. هدف این نسخه تحقیق، replay، ارزیابی و یادگیری کنترل‌شده است.

## ۱. اجرای یک‌کلیکی

بعد از راه‌اندازی اولیه، کار روزمره فقط این است:

1. MT4 را باز کنید و روی حساب Demo لاگین باشید.
2. Docker Desktop اگر خواستید باز باشد؛ این نسخه برای اجرای محلی به Docker وابسته نیست.
3. روی `run.bat` دوبار کلیک کنید.
4. برای توقف، `stop.bat` را اجرا کنید.

`run.bat` خودش محیط Python، `.env`، توکن داشبورد، وضعیت feed و سرور ASTRA را بررسی می‌کند و مرورگر را باز می‌کند.

## ۲. یک بار اتصال MT4

1. `mql4/ASTRAFeedEA.mq4` را در MetaEditor باز کنید.
2. Compile کنید.
3. آن را روی نمودار طلای موردنظر در MT4 Demo Attach کنید.
4. EA فایل‌های `ASTRA_MT4_FEED.json` و `ASTRA_MT4_M1.csv` را در `MetaQuotes\Terminal\Common\Files` می‌سازد.

این EA **read-only** است و هیچ تابعی برای بازکردن، بستن یا تغییر سفارش ندارد.

## ۳. تست Feed

```text
python -m goldbot mt4check
```

باید مشخصات حساب، Tick، Symbol و تعداد کندل‌ها را ببینید.

## ۴. تست داخلی

```text
python -m goldbot selftest
```

انتظار:

```text
ok: true
deterministic: true
leak_violations: 0
```

## ۵. حالت‌ها

- `SIMULATION`: داده بازار از MT4، ولی حساب مجازی و بدون سفارش واقعی.
- `DISABLED`: فقط تحلیل؛ حتی حساب مجازی هم اجرا نمی‌شود.

حالت پیش‌فرض `SIMULATION` است.

## ۶. Cent Account

پول حساب و قیمت بازار دو چیز جدا هستند. مثلاً `80000 USC` برای نمایش به `800 USD` تبدیل می‌شود؛ اما قیمت XAUUSD هرگز به‌خاطر Cent بودن تقسیم بر 100 نمی‌شود. مشخصات Tick، Contract Size، Volume و غیره از Feed گرفته می‌شوند.

## ۷. Learning

Champion در حین اجرای عادی خودش را تغییر نمی‌دهد. مسیر تغییر مدل:

`Experience → Evaluation → Candidate → Replay/OOS → Shadow → VALIDATED → Human Promotion`

اگر داده کافی نباشد، سیستم ادعای برتری یا احتمال کالیبره‌شده نشان نمی‌دهد.

## ۸. Backtest

برای داده تاریخی CSV:

```text
python -m goldbot backtest --csv XAUUSD_M1.csv --split 0.7
python -m goldbot ablation --csv XAUUSD_M1.csv
python -m goldbot replay --csv XAUUSD_M1.csv --verify
```

کمتر از ۳۰ معامله بسته‌شده به‌عنوان `INSUFFICIENT_DATA` علامت می‌خورد و معیارها به‌تنهایی evidence عملکرد محسوب نمی‌شوند.

## ۹. داشبورد

داشبورد ASTRA با ظاهر Mission Control فضایی طراحی شده است: ۹ کارت هوش، Forecast، Regime، Diversity، Memory، Learning، Replay، Safety Gate، Mission Log و Chart. کره و مدارها تزئینی‌اند و هرگز داده واقعی را جعل نمی‌کنند.

## ۱۰. محدودیت مهم

این پروژه تضمین سود یا پیش‌بینی قابل اعتماد بازار نمی‌دهد. مدل‌های Forecast ساده‌اند و عملکرد آن‌ها باید فقط با داده مستقل و واقعی ارزیابی شود.
