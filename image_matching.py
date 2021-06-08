import cv2
import numpy as np
import glob
print("Hay nhap duong dan: ")
path = input("Your image path: ")
original = cv2.imread(path)

# Sift and Flann
sift = cv2.xfeatures2d.SIFT_create()

kp_1, desc_1 = sift.detectAndCompute(original, None)

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Load all the images
all_images_to_compare = []
titles = []
rate_max=0
name=""
bb=[]

print("Hay nhap duong dan folder chua anh: ")
path = input("Your image folder path: ")

for f in glob.iglob(path+"\*"):
# for f in glob.iglob("images\*"):
    image = cv2.imread(f)
    titles.append(f)
    all_images_to_compare.append(image)
for image_to_compare, title in zip(all_images_to_compare, titles):

    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

    matches = flann.knnMatch(desc_1, desc_2, k=2)
    percentage_similarity = []
    good_points = []
    # good_points = []
    for m, n in matches:
        if m.distance <0.6*n.distance:
            good_points.append(m)

    number_keypoints = 0
    if len(kp_1) >= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)

    print("Title: " + title)
    # percentage_similarity= []
    rate= len((good_points)) / number_keypoints * 100
    percentage_similarity.append(rate)
    print("Similarity: " + str(int(rate)) + "\n")
    if rate >rate_max:
        rate_max=rate
        # name =image_to_compare.title()
        bb=image_to_compare


#print ("Max value perentage : ", max(percentage_similarity))
print ("Max value perentage : ", rate_max)
# anh=cv2.imread(name)
# cv2.imshow("Image", anh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print (bb)
kp_2, desc_2 = sift.detectAndCompute(bb, None)

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(desc_1, desc_2, k=2)
good_points = []

for m, n in matches:

    if m.distance < 0.6*n.distance:
        good_points.append(m)


result = cv2.drawMatches(original, kp_1, bb, kp_2, good_points, None)

cv2.imshow("original image", original)
cv2.imshow("fined Image", bb)
cv2.imshow("matched points", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

