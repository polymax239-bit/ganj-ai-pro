
GANJ AI LIVE PRO
================

ساختار فایل ها:
1. app.py
2. index.html
3. requirements.txt
4. Procfile
5. README_FA.txt

امکانات:
- اتصال به Binance Spot
- اتصال به Binance Futures
- دریافت کندل های بازار
- نمودار زنده
- دفتر سفارشات
- معاملات اخیر
- Open Interest
- Funding Rate
- Long / Short Ratio
- تحلیل EMA
- تحلیل RSI
- حمایت و مقاومت
- نمایش 10 کندل احتمالی آینده

راه اندازی:
1. فایل ها را در مخزن پروژه قرار دهید.
2. سرویس میزبانی را به مخزن متصل کنید.
3. دستور نصب:
   pip install -r requirements.txt
4. دستور اجرا:
   gunicorn app:app --bind 0.0.0.0:$PORT

CoinMarketCap:
برای دریافت اطلاعات بنیادی باید API Key معتبر
در متغیر محیطی CMC_API_KEY تنظیم شود.

نکته:
اطلاعات بازار از Binance دریافت می شود.
داده های دفتر سفارشات محدود به سطوح دریافتی API هستند.
پیش بینی کندل ها احتمالی است و تضمین سود نیست.
