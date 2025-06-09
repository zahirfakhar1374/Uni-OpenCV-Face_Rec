🧠 Face Detection with OpenCV & Image Navigation
این پروژه یک ابزار ساده و گرافیکی برای تشخیص چهره در تصاویر است که با استفاده از OpenCV توسعه داده شده و شامل قابلیت ورق‌زدن بین تصاویر (Next/Prev) با دکمه‌های گرافیکی است.

📸 ویژگی‌ها
تشخیص چهره با استفاده از Haar Cascade Classifier

مشاهده تصاویر موجود در پوشه‌ی images

دکمه‌های گرافیکی برای نمایش تصویر بعدی و قبلی

پیام مناسب در صورت نبود تصویر یا عدم تشخیص چهره

📁 ساختار پوشه
css
Copy
Edit
project/
│
├── haarcascade_frontalface_default.xml
├── main.py
├── images/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── README.md
🧰 پیش‌نیازها
برای اجرای این برنامه نیاز به نصب کتابخانه‌های زیر دارید:

bash
Copy
Edit
pip install opencv-python
▶️ اجرای برنامه
مطمئن شوید فایل haarcascade_frontalface_default.xml در کنار اسکریپت شما قرار دارد.
اگر ندارید می‌توانید از لینک زیر دانلود کنید:
دانلود Haarcascade

تصاویر خود را داخل پوشه images قرار دهید.

برنامه را اجرا کنید:

bash
Copy
Edit
python main.py
از دکمه‌های prev و next برای جابجایی بین تصاویر استفاده کنید.

⌨️ میان‌برها
q : خروج از برنامه

📷 نمونه خروجی
<div align="center"> <img src="https://upload.wikimedia.org/wikipedia/commons/2/22/Face_detection_using_OpenCV.png" width="500" /> </div>
📌 نکات
فقط فرمت‌های .jpg, .jpeg, .png پشتیبانی می‌شوند.

برای دیدن بهتر دکمه‌ها و عملکرد درست آن‌ها، وضوح تصویر به صورت ثابت (600×400) تنظیم شده است.

