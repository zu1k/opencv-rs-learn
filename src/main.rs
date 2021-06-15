use opencv::{
    core::{self, Scalar, Size, Point}, 
    imgcodecs, 
    imgproc, 
    prelude::*
};

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

    imgcodecs::imwrite(dst, &img_select, &core::Vector::default())?;
    Ok(())
}

fn main() {
    process("E:\\projects\\cv\\1.jpg", "E:\\projects\\cv\\2.jpg").unwrap()
}