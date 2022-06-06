import tensorflow as tf
import numpy as np

from tkinter import *
import os
from tkinter import filedialog
import cv2
import argparse, sys, os
import time
from matplotlib import pyplot as plt
from tkinter import messagebox


def endprogram():
	print ("\nProgram terminated!")
	sys.exit()


def file_sucess():
    global file_success_screen
    file_success_screen = Toplevel(training_screen)
    file_success_screen.title("File Upload Success")
    file_success_screen.geometry("150x100")
    file_success_screen.configure(bg='pink')
    Label(file_success_screen, text="File Upload Success").pack()
    Button(file_success_screen, text='''ok''', font=(
        'Verdana', 15), height="2", width="30").pack()

global ttype

def training():
    global training_screen

    global clicked

    training_screen = Toplevel(main_screen)
    training_screen.title("Training")
    # login_screen.geometry("400x300")
    training_screen.geometry("600x450+650+150")
    training_screen.minsize(120, 1)
    training_screen.maxsize(1604, 881)
    training_screen.resizable(1, 1)
    training_screen.configure()
    # login_screen.title("New Toplevel")



    Label(training_screen, text='''Upload Image ''', background="#d9d9d9", disabledforeground="#a3a3a3",
          foreground="#000000",  width="300", height="2", font=("Calibri", 16)).pack()
    Label(training_screen, text="").pack()


    options = [
        'Blotch_Apple',
                   'Cauliflower_Alternaria_Leaf_Spot',
                   'cauliflower_bacterial_soft_rot',
'cauliflower_black_leg',
'Citrus_Black_spot',
'Citrus_Canker',
'Citrus_Greening',
'Citrus_healthy',
'Citrus_Scab',
'Healthy_lemons',
'Lemon_canker',
'Lemon_mold',
'Lemon_scab',
'Normal_Apple',
'Rot_Apple',
'Scab_Apple'

    ]

    # datatype of menu text
    clicked = StringVar()


    # initial menu text
    clicked.set("Blotch_Apple")

    # Create Dropdown menu
    drop = OptionMenu(training_screen, clicked, *options )
    drop.config(width="30")

    drop.pack()

    ttype=clicked.get()

    Button(training_screen, text='''Upload Image''', font=(
        'Verdana', 15), height="2", width="30", command=imgtraining).pack()


def imgtraining():
    name1 = clicked.get()

    print(name1)

    import_file_path = filedialog.askopenfilename()
    import os
    s = import_file_path
    os.path.split(s)
    os.path.split(s)[1]
    splname = os.path.split(s)[1]


    image = cv2.imread(import_file_path)
    #filename = 'Test.jpg'
    filename = 'Data/'+name1+'/'+splname

    cv2.imwrite(filename, image)
    print("After saving image:")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Original image', image)
    cv2.imshow('Gray image', gray)
    # import_file_path = filedialog.askopenfilename()
    print(import_file_path)
    fnm = os.path.basename(import_file_path)
    print(os.path.basename(import_file_path))

    from PIL import Image, ImageOps

    im = Image.open(import_file_path)
    im_invert = ImageOps.invert(im)
    im_invert.save('lena_invert.jpg', quality=95)
    im = Image.open(import_file_path).convert('RGB')
    im_invert = ImageOps.invert(im)
    im_invert.save('tt.png')
    image2 = cv2.imread('tt.png')
    cv2.imshow("Invert", image2)

    """"-----------------------------------------------"""

    img = image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Original image', img)
    cv2.imshow('Gray image', gray)
    #dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    dst = cv2.medianBlur(img, 7)
    cv2.imshow("Nosie Removal", dst)


def fulltraining():
    import model as mm


def testing():
    global testing_screen
    testing_screen = Toplevel(main_screen)
    testing_screen.title("Testing")
    # login_screen.geometry("400x300")
    testing_screen.geometry("600x450+650+150")
    testing_screen.minsize(120, 1)
    testing_screen.maxsize(1604, 881)
    testing_screen.resizable(1, 1)
    testing_screen.configure()
    # login_screen.title("New Toplevel")

    Label(testing_screen, text='''Upload Image''', disabledforeground="#a3a3a3",
          foreground="#000000", width="300", height="2",bg='pink', font=("Calibri", 16)).pack()
    Label(testing_screen, text="").pack()
    Label(testing_screen, text="").pack()
    Label(testing_screen, text="").pack()
    Button(testing_screen, text='''Upload Image''', font=(
        'Verdana', 15), height="2", width="30", command=imgtest).pack()


global affect
def imgtest():

    import_file_path = filedialog.askopenfilename()

    image = cv2.imread(import_file_path)
    print(import_file_path)
    filename = 'Output/Out/Test.jpg'
    cv2.imwrite(filename, image)
    print("After saving image:")
    #result()

    img = cv2.imread(import_file_path)
    if img is None:
        print('no data')

    img1 = cv2.imread(import_file_path)
    print(img.shape)
    img = cv2.resize(img, ((int)(img.shape[1]), (int)(img.shape[0])))
    original = img.copy()
    neworiginal = img.copy()

    cv2.imshow('original', img1)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Original image', img1)

    cv2.imshow('Gray image', gray)
    p = 0
    for i in range(img.shape[0]):

        for j in range(img.shape[1]):
            B = img[i][j][0]
            G = img[i][j][1]
            R = img[i][j][2]
            if (B > 110 and G > 110 and R > 110):
                p += 1

    totalpixels = img.shape[0] * img.shape[1]
    per_white = 100 * p / totalpixels
    if per_white > 10:
        img[i][j] = [500, 300, 200]

        cv2.imshow('color change', img)
    # Guassian blur
    blur1 = cv2.GaussianBlur(img, (3, 3), 1)
    # mean-shift algo
    newimg = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    img = cv2.pyrMeanShiftFiltering(blur1, 20, 30, newimg, 0, criteria)
    cv2.imshow('means shift image', img)
    # Guassian blur
    blur = cv2.GaussianBlur(img, (11, 11), 1)
    # Canny-edge detection
    canny = cv2.Canny(blur, 160, 290)
    canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    # contour to find leafs
    bordered = cv2.cvtColor(canny, cv2.COLOR_BGR2GRAY)
    _,contours, hierarchy = cv2.findContours(bordered, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    maxC = 0
    for x in range(len(contours)):
        if len(contours[x]) > maxC:
            maxC = len(contours[x])
            maxid = x
    perimeter = cv2.arcLength(contours[maxid], True)
    # print perimeter
    Tarea = cv2.contourArea(contours[maxid])
    cv2.drawContours(neworiginal, contours[maxid], -1, (0, 0, 255))
    cv2.imshow('Contour', neworiginal)
    # cv2.imwrite('Contour complete leaf.jpg',neworiginal)
    # Creating rectangular roi around contour
    height, width, _ = canny.shape
    min_x, min_y = width, height
    max_x = max_y = 0
    frame = canny.copy()
    # computes the bounding box for the contour, and draws it on the frame,
    for contour, hier in zip(contours, hierarchy):
        (x, y, w, h) = cv2.boundingRect(contours[maxid])
        min_x, max_x = min(x, min_x), max(x + w, max_x)
        min_y, max_y = min(y, min_y), max(y + h, max_y)
        if w > 80 and h > 80:
            # cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)   #we do not draw the rectangle as it interferes with contour later on
            roi = img[y:y + h, x:x + w]
            originalroi = original[y:y + h, x:x + w]
    if (max_x - min_x > 0 and max_y - min_y > 0):
        roi = img[min_y:max_y, min_x:max_x]
        originalroi = original[min_y:max_y, min_x:max_x]
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0),
                      2)  # we do not draw the rectangle as it interferes with contour
    cv2.imshow('ROI', frame)
    cv2.imshow('rectangle ROI', roi)
    img = roi
    # Changing colour-space
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imghls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
    cv2.imshow('HLS', imghls)
    imghls[np.where((imghls == [30, 200, 2]).all(axis=2))] = [0, 200, 0]
    cv2.imshow('new HLS', imghls)
    # Only hue channel
    huehls = imghls[:, :, 0]
    cv2.imshow('img_hue hls', huehls)
    # ret, huehls = cv2.threshold(huehls,2,255,cv2.THRESH_BINARY)
    huehls[np.where(huehls == [0])] = [35]
    cv2.imshow('img_hue with my mask', huehls)
    # Thresholding on hue image
    ret, thresh = cv2.threshold(huehls, 28, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('thresh', thresh)
    # Masking thresholded image from original image
    mask = cv2.bitwise_and(originalroi, originalroi, mask=thresh)
    cv2.imshow('masked out img', mask)
    # Finding contours for all infected regions
    _,contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    Infarea = 0
    for x in range(len(contours)):
        cv2.drawContours(originalroi, contours[x], -1, (0, 0, 255))
        cv2.imshow('Contour masked', originalroi)
        # Calculating area of infected region
        Infarea += cv2.contourArea(contours[x])

    if Infarea > Tarea:
        Tarea = img.shape[0] * img.shape[1]

    print('_________________________________________\n Perimeter: %.2f' % (perimeter)
          + '\n_________________________________________')
    print('_________________________________________\n Total area: %.2f' % (Tarea)
          + '\n_________________________________________')
    # Finding the percentage of infection in the leaf
    print('_________________________________________\n Infected area: %.2f' % (Infarea)
          + '\n_________________________________________')
    try:
        per = 100 * Infarea / Tarea
    except ZeroDivisionError:
        per = 0
    print('_________________________________________\n Percentage of infection region: %.2f' % (per)
          + '\n_________________________________________')
    print("\n*To terminate press and hold (q)*")
    cv2.imshow('orig', original)

    result()


def result():
    import warnings
    warnings.filterwarnings('ignore')

    import tensorflow as tf
    classifierLoad = tf.keras.models.load_model('model.h5')

    import numpy as np
    from keras.preprocessing import image

    test_image = image.load_img('Output/Out/Test.jpg', target_size=(200, 200))
    img1 = cv2.imread('Output/Out/Test.jpg')
    # test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifierLoad.predict(test_image)

    out = ''
    pre=''
    if result[0][0] == 1:
        print("Blotch_Apple")
        out="Blotch_Apple"
        pre ="Sprays of strobilurin fungicide, kresxim methyl or trifloxystrobin"
    elif result[0][1] == 1:
        print("Cauliflower_Alternaria_Leaf_Spot")
        out="Cauliflower_Alternaria_Leaf_Spot"
        pre = "Fungicides are available to help control Alternaria leaf spot include chlorothalonil and azoxystrobin"
    elif result[0][2] == 1:
        print("cauliflower_bacterial_soft_rot")
        out = "cauliflower_bacterial_soft_rot"
        pre = "Clean and spray storage walls and floors with copper sulfate solution (1 lb/5 gal water). Bactericides such as Clorox, Lysol, and quaternary ammonium products"
    elif result[0][3] == 1:
        print("cauliflower_black_leg")
        out = "cauliflower_black_leg"
        pre = "Rovral pesticides are most effective when used in combination with cultural control strategies."
    elif result[0][4] == 1:
        print("Citrus_Black_spot")
        out = "Citrus_Black_spot"
        pre = "Copper-based sprays alone or together with an antibiotic or the chemical mancozeb can be used with moderate efficacy."
    elif result[0][5] == 1:
        print("Citrus_Canker")
        out = "Citrus_Canker"
        pre = " Liquid copper fungicide sprays can be effective in managing citrus canker and control to a limit"
    elif result[0][6] == 1:
        print("Citrus_Greening")
        out = "Citrus_Greening"
        pre = "Imazalil, thiabenadazole fungiside is effective to control citrus greening disease"
    elif result[0][7] == 1:
        print("Citrus_healthy")
        out = "Citrus_healthy"
        pre = ""
    elif result[0][8] == 1:
        print("Citrus_Scab")
        out = "Citrus_Scab"
        pre = "Protectant fungicides based thiram, difenoconazole and chlorothalonil can be used preventively to avoid a widespread infection."
    elif result[0][9] == 1:
        print("Healthy_lemons")
        out = "Healthy_lemons"
        pre = ""
    elif result[0][10] == 1:
        print("Lemon_canker")
        out = "Lemon_canker"
        pre = "Treat affected plants with Yates Liquid Copper after the removal of dead or affected tissue. Healthy plants are better at resisting"
    elif result[0][11] == 1:
        print("Lemon_mold")
        out = "Lemon_mold"
        pre = "initially control the insect outbreak and then splash the lemon tree with Neem oil insecticide spray, both the sides of the leaves. Repeat the same after 2 weeks, contingent on the degree of the invasion. At last treat the mold growth with copper fungicide."
    elif result[0][12] == 1:
        print("Lemon_scab")
        out = "Lemon_scab"
        pre = "Protectant fungicides based thiram, difenoconazole and chlorothalonil can be used preventively to avoid a widespread infection."
    elif result[0][13] == 1:
        print("Normal_Apple")
        out = "Normal_Apple"
        pre = ""
    elif result[0][14] == 1:
        print("Rot_Apple")
        out = "Rot_Apple"
        pre = "broad-spectrum fungicide, like captan, metiram (Polyram), mancozeb, Ziram, thiram, sulfur, or ferbam."
    elif result[0][15] == 1:
        print("Scab_Apple")
        out = "Scab_Apple"
        pre = "Cultural practices and fungicides can help control sooty blotch and flyspeck."


    messagebox.showinfo("Result", "Classification Result : "+str(out))
    messagebox.showinfo("Pesticide ", "Pesticide  : " + str(pre))



def main_account_screen():
    global main_screen
    main_screen = Tk()
    width = 600
    height = 600
    screen_width = main_screen.winfo_screenwidth()
    screen_height = main_screen.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    main_screen.geometry("%dx%d+%d+%d" % (width, height, x, y))
    main_screen.resizable(0, 0)
    # main_screen.geometry("300x250")
    main_screen.configure()
    main_screen.title(" Fruit Disease Prediction")

    Label(text=" Fruit Disease Prediction", width="300", height="5", font=("Calibri", 16)).pack()

    Button(text="UploadImage", font=(
        'Verdana', 15), height="2", width="30", command=training, highlightcolor="black").pack(side=TOP)
    Label(text="").pack()
    Button(text="Training", font=(
        'Verdana', 15), height="2", width="30", command=fulltraining, highlightcolor="black").pack(side=TOP)

    Label(text="").pack()
    Button(text="Testing", font=(
        'Verdana', 15), height="2", width="30", command=testing).pack(side=TOP)

    Label(text="").pack()

    main_screen.mainloop()

main_account_screen()