#!/usr/bin/python
#  -*- coding: utf-8 -*-

import argparse
from subprocess import call
from os.path import join, basename, splitext, isdir, isfile, dirname, abspath
from os import makedirs
from deps import opencv_bin_dir, project_root_dir


parser = argparse.ArgumentParser(description="Extracts positive samples from a single image applying random geometric "
                                             "and photometric transforms. Trains a cascade classifier.")
parser.add_argument("--image", type=str, required=True, help="image with object of interest")
parser.add_argument("--features", type=str, default="LBP", help="features to be used in cascade detector (HAAR or LBP)")
parser.add_argument("--width", type=int, default=32, help="width of the object image")
parser.add_argument("--height", type=int, default=32, help="height of the object image")
parser.add_argument("--out", type=str, default="", help="output model name")
parser.add_argument("--show", action='store_true', help="whether to show augmented positive samples")
args = parser.parse_args()

create_samples_app = "opencv_createsamples.exe"
train_cascade_app = "opencv_traincascade.exe"

base_name = basename(args.image)
base_name = splitext(base_name)[0]
if not args.out == "":
	base_name = args.out
vec_file_path = base_name + ".vec"
negatives_list_file = join(project_root_dir, "images", "negatives.txt")

create_samples_call = '"{0}" -img "{1}" -bgcolor 255 -bgthresh 0 -bg "{2}" -maxxangle 0.9 -maxyangle 0.9 -maxzangle 3.14 ' \
                      '-num 1000 -w {3} -h {4} -vec "{5}"'.format(join(opencv_bin_dir, create_samples_app),
                                                                  args.image, negatives_list_file, args.width,
                                                                  args.height, vec_file_path)
if args.show:
	create_samples_call += " -show"
	print("Press <ESC> to continue w\o waiting for key pressure")

print("call: " + create_samples_call)
call(create_samples_call, shell=True)

cascade_dir = base_name + "_cascade"
if not isdir(cascade_dir):
    makedirs(cascade_dir)
train_cascade_call = '"{0}" -vec "{1}" -numPos 500 -bg "{2}" -numNeg 500 -featureType {3} -maxFalseAlarmRate 0.1 ' \
                     '-numStages 4 -w {4} -h {5} -data "{6}"'.format(join(opencv_bin_dir, train_cascade_app),
                                                                     vec_file_path, negatives_list_file, args.features,
                                                                     args.width, args.height, cascade_dir)
print("call: " + train_cascade_call)
call(train_cascade_call, shell=True)

detector_file = join(cascade_dir, "cascade.xml")
if isfile(detector_file):
    print("cascade detector has been successfully trained and located at '{}'.".format(detector_file))
