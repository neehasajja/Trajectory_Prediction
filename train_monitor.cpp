#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <cassert>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void cvae_training_monitor(const vector<double>& mse_losses, const vector<double>& vlb_losses, string criterion) {
    Mat x_axis(1, mse_losses.size(), CV_64F);
    iota(x_axis.begin<double>(), x_axis.end<double>(), 0);

    Mat total_loss(1, mse_losses.size(), CV_64F);
    for (int i = 0; i < mse_losses.size(); i++) {
        total_loss.at<double>(i) = mse_losses[i] + vlb_losses[i];
    }

    Mat mse(1, mse_losses.size(), CV_64F);
    mse = Mat(mse_losses).t();

    Mat vlb(1, vlb_losses.size(), CV_64F);
    vlb = Mat(vlb_losses).t();

    vector<Mat> plot_data = { total_loss, mse, vlb };

    vector<string> labels = { "MSE + " + criterion + " loss", "MSE", criterion + " loss" };

    int plot_cols = 3;
    int plot_rows = 1;

    int image_width = 480 * plot_cols;
    int image_height = 320 * plot_rows;

    Mat image(image_height, image_width, CV_8UC3, Scalar(255, 255, 255));
    int x = 30;
    int y = 20;
    int plot_width = 400;
    int plot_height = 260;

    for (int i = 0; i < plot_cols; i++) {
        Rect roi(x + i * plot_width, y, plot_width, plot_height);
        Mat current_plot(image, roi);

        Mat y_axis;
        minMaxLoc(plot_data[i], &y_axis.at<double>(0), &y_axis.at<double>(1));
        y_axis.at<double>(0) = 0.9 * y_axis.at<double>(0);
        y_axis.at<double>(1) = 1.1 * y_axis.at<double>(1);

        Mat plot_image = Mat::zeros(plot_height, plot_width, CV_8UC3);
        Mat x_mapper = (Mat_<double>(2, 2) << 0, plot_width, y_axis.at<double>(0), y_axis.at<double>(1));
        Mat y_mapper = (Mat_<double>(2, 2) << 0, plot_height, 0, 1);
        Mat mapping = x_mapper * y_mapper.inv();

        Mat plot_data_mapped;
        perspectiveTransform(plot_data[i], plot_data_mapped, mapping
