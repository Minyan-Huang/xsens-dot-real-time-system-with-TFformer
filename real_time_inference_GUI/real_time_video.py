import cv2

def main():
    # 開啟USB攝像頭（設備索引0，若有多個攝像頭可以換成1、2等）
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("無法開啟攝像頭")
        return

    while True:
        # 從攝像頭讀取一幀
        ret, frame = cap.read()

        if not ret:
            print("無法接收影像（stream end?）")
            break

        # 顯示影像
        cv2.imshow('frame', frame)

        # 若按下q鍵則退出循環
        if cv2.waitKey(1) == ord('q'):
            break

    # 釋放攝像頭並關閉所有OpenCV視窗
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
