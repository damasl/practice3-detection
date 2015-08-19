#include <iostream>
#include <string>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

using namespace std;
using namespace cv;

const char* params =
     "{ h | help     | false | print usage                                   }"
     "{   | detector |       | XML file with a cascade detector              }"
     "{   | image    |       | image to detect objects on                    }"
     "{   | video    |       | video file to detect on                       }"
     "{   | camera   | false | whether to detect on video stream from camera }";


void drawDetections(const vector<Rect>& detections,
                    const Scalar& color,
                    Mat& image)
{
    for (size_t i = 0; i < detections.size(); ++i)
    {
        rectangle(image, detections[i], color, 2);
    }
}

const Scalar red(0, 0, 255);
const Scalar green(0, 255, 0);
const Scalar blue(255, 0, 0);
const Scalar colors[] = {red, green, blue};

int main(int argc, char** argv)
{
    // Parse command line arguments.
    CommandLineParser parser(argc, argv, params);
    // If help flag is present, print help message and exit.
    if (parser.get<bool>("help"))
    {
        parser.printParams();
        return 0;
    }

    string detector_file = parser.get<string>("detector");
    CV_Assert(!detector_file.empty());
	std::cout<<detector_file;
    string image_file = parser.get<string>("image");
    string video_file = parser.get<string>("video");
    bool use_camera = parser.get<bool>("camera");

    // TODO: Load detector.

    if (!image_file.empty())
    {
		Mat src = imread(image_file);
		vector<Rect> objects;
		CascadeClassifier cascade;
		cascade.load(detector_file);
		cascade.detectMultiScale(src, objects);
		drawDetections(objects, Scalar(10,10,10), src);
		imshow("image",src);
		waitKey();
    }
    else if (!video_file.empty())
    {
		CascadeClassifier cascade;
		cascade.load(detector_file);
        VideoCapture cap(video_file);
		Mat src;
		if (cap.isOpened())
		for(;;)
		{
			if (cap.read(src)){
				vector<Rect> objects;
				cascade.detectMultiScale(src, objects);
				drawDetections(objects, red, src);
				imshow("video",src);
				waitKey(1);
			}
			else break;
		}
    }
    else if (use_camera)
    {
        VideoCapture cap(0);
		//Нулевой кадр
		Mat src;
		cap.read(src);
		for(;;)
		{
			if (!cap.isOpened()) {std::cout<<"cap is not opened\n"; return 0;}
			cap.read(src);
			imshow("camera",src);
			vector<Rect> objects;
			CascadeClassifier cascade;
			cascade.load(detector_file);
			cascade.detectMultiScale(src, objects);
			drawDetections(objects, Scalar(0,0,0), src);
			
			if(waitKey('q') >= 0) break;
			
		}
		

    }
    else
    {
        cout << "Declare a source of images to detect on." << endl;
    }

    return 0;
}



