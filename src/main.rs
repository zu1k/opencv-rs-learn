use opencv::{core::{self, CV_32F, Point, Scalar, Size}, imgcodecs, imgproc, prelude::*};

fn process(src: &str, dst: &str) -> opencv::Result<()> {
    let img_origin = imgcodecs::imread(src, imgcodecs::IMREAD_ANYCOLOR)?;
    
    // convert to HSV
    let mut img_hsv = Mat::default();
    imgproc::cvt_color(
        &img_origin, 
        &mut img_hsv, 
        imgproc::COLOR_BGR2HSV , 
        0
    )?;

    // red range 1
    let mut img_range1 = Mat::default();
    core::in_range(&img_hsv, 
        &Scalar::new(0., 43., 46., 0.), 
        &Scalar::new(10., 255., 255., 0.), 
        &mut img_range1
    )?;
    
    // red range 2
    let mut img_range2 = Mat::default();
    core::in_range(
        &img_hsv, 
        &Scalar::new(156., 43., 46., 0.), 
        &Scalar::new(180., 255., 255., 0.), 
        &mut img_range2
    )?;

    // red range
    let mut img_range = Mat::default();
    core::add(
        &img_range1, 
        &img_range2, 
        &mut img_range, 
        &core::no_array()?, 
        -1
    )?;

    // dilate
    let mut img_select = Mat::default();
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE, 
        Size::new(3, 3), 
        Point::new(-1, -1)
    )?;
    imgproc::dilate(
        &img_range, 
        &mut img_select, &kernel, 
        Point::new(-1, -1), 
        2, 
        core::BORDER_CONSTANT, 
        imgproc::morphology_default_border_value()?
    )?;

    // 设置黑色区域为透明
    let mut img_trans = Mat::default();
    imgproc::cvt_color(
        &img_select, 
        &mut img_trans, 
        imgproc::COLOR_BGR2BGRA, 
        0
    )?;

    

    imgcodecs::imwrite(dst, &img_select, &core::Vector::default())?;
    Ok(())
}

fn a4(src: &str) -> opencv::Result<()> {
    let height = 900;

    // read img file
    let img_origin = imgcodecs::imread(src, imgcodecs::IMREAD_ANYCOLOR)?;

    // resize
    let mut img_resize = Mat::default();
    let devide = img_origin.size()?.height/height;
    let width = img_origin.size()?.width/devide;
    imgproc::resize(
        &img_origin, 
        &mut img_resize, 
        core::Size::new(width, height),
        0., 
        0., 
        imgproc::INTER_LINEAR
    )?;

    // gaussian blur
    let mut img_gauss = Mat::default();
    imgproc::gaussian_blur(
        &img_resize, 
        &mut img_gauss, 
        Size::new(3, 3), 
        2., 
        2., 
        core::BORDER_DEFAULT
    )?;

    // canny
    let mut img_canny = Mat::default();
    imgproc::canny(
        &img_gauss, 
        &mut img_canny, 
        60., 
        240., 
        3, 
        false
    )?;

    // dilate
    let mut img_dilate = Mat::default();
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_ELLIPSE, 
        Size::new(3, 3), 
        Point::new(-1, -1)
    )?;
    imgproc::dilate(
        &img_canny, 
        &mut img_dilate, 
            &kernel, 
        Point::new(-1, -1), 
        1, 
        core::BORDER_CONSTANT, 
        imgproc::morphology_default_border_value()?
    )?;

    // find max contour
    let mut contours: core::Vector<core::Vector<Point>> = core::Vector::default();
    imgproc::find_contours(
        &img_dilate,
        &mut contours, 
        imgproc::RETR_EXTERNAL, 
        imgproc::CHAIN_APPROX_NONE, 
        Point::new(-1, -1)
    )?;
    let mut max_contour: core::Vector<Point> = core::Vector::default();
    let mut max_area = 0.;
    for contour in contours {
        let area = imgproc::contour_area(&contour, false)?;
        if area>max_area {
            max_area = area;
            max_contour = contour;
        }
    }

    // get box point
    let mut hull = core::Vector::<Point>::default();
    imgproc::convex_hull(
        &max_contour, 
        &mut hull, 
        false, 
        true
    )?;

    let epsilon = 0.02 * imgproc::arc_length(&max_contour, true)?;
    let mut approx = core::Vector::<Point>::default();
    imgproc::approx_poly_dp(
        &hull, 
        &mut approx, 
        epsilon, 
        true
    )?;

    let ratio:f64 = 900.0/img_origin.size()?.height as f64;
    // // ada point
    let mut points = Mat::default();
    if ratio!=1.0 {
        points = Mat::from_exact_iter(approx.iter().map(|x| x/ratio as i32))?;
    } else {
        points = Mat::from_exact_iter(approx.iter())?;
    }
    let boxes = points.reshape(points.rows(), 2)?;

    // order






    Ok(())
}


fn main() {
    // process("E:\\projects\\cv\\1.jpg", "E:\\projects\\cv\\2.jpg").unwrap()
    a4("E:\\projects\\cv\\1.jpg").unwrap();
}