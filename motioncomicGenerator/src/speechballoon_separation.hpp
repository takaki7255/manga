//
//  speechballoon_separation.hpp
//  main
//
//  Created by 田中海斗 on 2022/11/29.
//

#ifndef speechballoon_separation_hpp
#define speechballoon_separation_hpp

#include <vector>
#include <stdio.h>
#include "opencv2/opencv.hpp"
class Speechballoon
{
    public:
        std::vector<cv::Mat> speechballoon_detect(cv::Mat &src_img);//コマ画像
    
    private:
};
#endif /* speechballoon_separation_hpp */
