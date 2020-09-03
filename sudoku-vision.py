import argparse
from cv2 import cv2
from extractGrid import solve_and_print

# setting up parser
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='', help='Input file (eg: testpic.jpg)')
args = parser.parse_args()

def video_mode():
    video = cv2.VideoCapture(0)

    while(video.isOpened()):
        ret, frame = video.read()
        if ret == True:
            solved = solve_and_print(frame)
            cv2.imshow("Video screen",solved)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()

def image_mode(filename):
    image = cv2.imread(args.file)
    if image is not None:
        solved = solve_and_print(image)
        cv2.imshow("image",image)
        cv2.waitKey(0)
    else:
        raise IOError('Image not found')


def main():

    # if the user has not specified a file to load, use the video input
    if args.file == '':
        video_mode()
    else:
        image_mode(args.file)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()