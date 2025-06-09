import cv2
import os

# مسیر فایل Haar Cascade
cascade_path = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# پوشه عکس‌ها
image_folder = 'images'

# لیست عکس‌ها
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
image_files.sort()

if not image_files:
    print("هیچ عکسی در پوشه پیدا نشد.")
    exit()

fixed_size = (600, 400)
button_height = 50
button_width = 100
button_color = (50, 150, 50)
button_hover_color = (70, 200, 70)
text_color = (255, 255, 255)

index = 0
window_name = 'Face Detection'

# متغیر برای ذخیره وضعیت دکمه ها (hover یا نه)
buttons = {
    'prev': {'pos': (50, fixed_size[1] + 10, button_width, button_height)},
    'next': {'pos': (fixed_size[0] - 150, fixed_size[1] + 10, button_width, button_height)},
}

clicked_button = None

def draw_buttons(img):
    mouse_pos = cv2.getWindowImageRect(window_name)[:2]
    # (نمی‌توانیم موس رو مستقیما بگیریم اینجا)
    # بنابراین فقط دکمه‌ها رو رسم می‌کنیم بدون hover برای ساده بودن

    for btn_name, btn in buttons.items():
        x, y, w, h = btn['pos']
        cv2.rectangle(img, (x, y), (x + w, y + h), button_color, -1)
        text = 'prev' if btn_name == 'prev' else 'next'
        cv2.putText(img, text, (x + 20, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

# تابعی که موقع کلیک ماوس اجرا می‌شود
def mouse_callback(event, x, y, flags, param):
    global index

    if event == cv2.EVENT_LBUTTONDOWN:
        for btn_name, btn in buttons.items():
            bx, by, bw, bh = btn['pos']
            if bx <= x <= bx + bw and by <= y <= by + bh:
                if btn_name == 'prev':
                    index = (index - 1) % len(image_files)
                elif btn_name == 'next':
                    index = (index + 1) % len(image_files)

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

while True:
    image_name = image_files[index]
    img_path = os.path.join(image_folder, image_name)
    image = cv2.imread(img_path)
    if image is None:
        print(f"نمی‌توانم عکس {image_name} را باز کنم.")
        index = (index + 1) % len(image_files)
        continue

    image = cv2.resize(image, fixed_size)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # اضافه کردن فضای پایین برای دکمه‌ها
    img_with_buttons = cv2.copyMakeBorder(image, 0, button_height + 20, 0, 0, cv2.BORDER_CONSTANT, value=[50, 50, 50])
    draw_buttons(img_with_buttons)

    cv2.imshow(window_name, img_with_buttons)

    key = cv2.waitKey(30)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
