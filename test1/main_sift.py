import cv2
# pip install opencv-python
# pip install opencv-contrib-python
# Open images
img1 = cv2.imread('./images/adobo_flank_steak_-_admony-mia.jpg')
img2 = cv2.imread('./images/avocadochickentacos-ruben-mia.jpg')

#img1 = cv2.imread('./images/people/1.jpeg')
#img2 = cv2.imread('./images/people/2.jpeg')

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Sift characteristics
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# Compare
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good_matches = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good_matches.append([m])

# Draw results
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('Resultado', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()