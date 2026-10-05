# گزارش بررسی ASTRA TRADER — نسخه MT4 Read-Only / Simulation

این نسخه برای رفع مشکل راه‌اندازی و خطای Compile در EA بررسی شد.

## اصلاح‌های اصلی

- خطای MQL4 مربوط به `AccountMarginLevel()` حذف شد. سطح مارجین داخل EA از رابطه Equity / Margin × 100 محاسبه می‌شود تا با محیط MQL4 شما سازگار باشد.
- `run.bat` اکنون فقط Python `bootstrap.py` را اجرا می‌کند؛ تمام argument/process handling در Python انجام می‌شود تا خطاهای parser ویندوز و اجرای ناخواسته Python REPL رخ ندهد.
- launcher دیگر جست‌وجوی بازگشتی و سنگین داخل کل `Program Files` انجام نمی‌دهد.
- launcher حالت اجرای پروژه را روی `SIMULATION` و منبع داده را روی `MT4` نگه می‌دارد.
- اگر Feed در دسترس نباشد، launcher پس از timeout خطای واضح و مسیر دقیق فایل مورد انتظار را نشان می‌دهد.
- `stop.bat` به‌صورت امن از Python داخل `.venv` استفاده می‌کند و به Python سراسری وابسته نیست.
- هشدار synthetic backtest به stderr منتقل شد تا خروجی stdout برای JSON قابل پردازش باشد.
- در EA هیچ API ارسال/ویرایش/بستن سفارش وجود ندارد.

## آزمون‌های انجام‌شده در محیط Linux

- Python `compileall`: PASS
- تمام 167 تست خارج از `test_engine.py`: **167 passed**
- `TestEngineSimulation`: **11 passed**
- `TestEngineSafety`: **7 passed**
- مجموع تست‌های اجراشده به‌صورت ماژولی: **185 passed**
- `goldbot selftest`: `ok=true`, `deterministic=true`, `leak_violations=0`
- backtest مصنوعی 8000 کندل M1 اجرا شد؛ نتیجه با `evaluation_status=INSUFFICIENT_DATA` و حداقل 30 معامله برای اعتبار آماری گزارش شد.
- health endpoint داشبورد در demo: HTTP 200
- درخواست snapshot بدون توکن: HTTP 401 (رفتار امنیتی مورد انتظار)

## تستی که فقط روی کامپیوتر کاربر قابل انجام است

MetaEditor و خود MT4 در این محیط لینوکس نصب نیستند؛ بنابراین Compile واقعی `ASTRAFeedEA.mq4` داخل MetaEditor ویندوز قابل تأیید مستقیم نبود. من خطای مشاهده‌شده را از کد حذف کردم و تمام توابع/ساختار EA باقی‌مانده را به‌صورت static بررسی کردم.

بعد از Compile موفق در MetaEditor باید پایین پنجره `0 errors` دیده شود.
