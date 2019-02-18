package com.plansolve.farm.util;

import com.plansolve.farm.model.ImageViewer;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import java.util.*;
import java.io.File;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;

import static org.opencv.imgproc.Imgproc.line;
import static org.opencv.imgproc.Imgproc.rectangle;

/**
 * @Author: Andrew
 * @Date: 2019/1/30
 * @Description:
 */
public class ImageUtil {

    private static final String folderUrl = "D:/usr/local/picture/";

    private static final float nndrRatio = 0.7f;//这里设置既定值为0.7，该值可自行调整

    //opencv各种函数的依赖库，必须导入！
    static{ System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main(String[] args) {
//        System.out.println("Welcome to OpenCV " + Core.VERSION);
//        Mat m = new Mat(5, 10, CvType.CV_8UC1, new Scalar(0));
//        System.out.println("OpenCV Mat: " + m);
//        Mat mr1 = m.row(1);
//        mr1.setTo(new Scalar(1));
//        Mat mc5 = m.col(5);
//        mc5.setTo(new Scalar(5));
//        System.out.println("OpenCV Mat data:\n" + m.dump());
        //模板匹配算法
//        String originalImg = "D:/usr/local/model/e57fc64deb1d42b98dded9a9e418b925.png";
//        String modelImg = "D:/usr/local/picture/6.jpg";
//        String targetImg = "D:/usr/local/picture/checkedImg.jpg";
//        modeltoPicture(modelImg, originalImg, targetImg);
        //特征提取匹配算法
//        String templateFilePath = "D:/usr/local/model/head.jpg";
//        String originalFilePath = "D:/usr/local/picture/head_img.jpg";
//        //图片灰度处理
//        Mat temImg = Imgcodecs.imread(templateFilePath);
//        Mat temGrayImage = new Mat(temImg.rows(),temImg.cols(),temImg.type());
//        Imgproc.cvtColor(temImg, temGrayImage, Imgproc.COLOR_RGB2GRAY);
//        Mat oriImg = Imgcodecs.imread(originalFilePath);
//        Mat originalGrayImage = new Mat(oriImg.rows(),oriImg.cols(),oriImg.type());
//        Imgproc.cvtColor(oriImg, originalGrayImage, Imgproc.COLOR_RGB2GRAY);
//        //图片比对
//        Integer points = getMatchImage(temGrayImage, originalGrayImage);
//        System.out.println("匹配的像素点总数：" + points);
        //读取并显示图片
//        Mat mat = Imgcodecs.imread("D:/usr/local/picture/mv.jpg");
//        ImageViewer imageViewer = new ImageViewer(mat, "第一幅图片");
//        imageViewer.imshow();

//        getFileList("D:/usr/local/picture/", null);

//        String path2 = ImageUtil.class.getResource("/").getPath();
//        System.out.println(path2);
        Mat src = Imgcodecs.imread("D:/usr/local/picture/mv.jpg");
        Imgproc.resize(src, src, new Size(50,50), 0, 0, Imgproc.INTER_AREA);
        Imgcodecs.imwrite("D:/usr/local/dnn/mv_dnn.jpg", src);

    }

    /**
     * 模板匹配算法-缺陷在于如果模板图片发生了旋转、缩放之后，这种通过滑窗的模板匹配方式就会失效
     * @param originalImg 原图路径
     * @param modelImg 模板图路径
     */
    public static void modeltoPicture(String originalImg, String modelImg, String targetImg){
        // 声明两个Mat变量
        Mat source, template;
        //将文件读入为OpenCV的Mat格式
        source = Imgcodecs.imread(originalImg);
        template = Imgcodecs.imread(modelImg);
        //创建于原图相同的大小，储存匹配度
        Mat result = Mat.zeros(source.rows() - template.rows() + 1, source.cols() - template.cols() + 1, CvType.CV_32FC1);
        //调用模板匹配方法
        Imgproc.matchTemplate(source, template, result, Imgproc.TM_SQDIFF_NORMED);
        //规格化
        Core.normalize(result, result, 0, 1, Core.NORM_MINMAX, -1);
        //获得最可能点，MinMaxLocResult是其数据格式，包括了最大、最小点的位置x、y
        Core.MinMaxLocResult mlr = Core.minMaxLoc(result);
        Point matchLoc = mlr.minLoc;
        //在原图上的对应模板可能位置画一个绿色矩形
        rectangle(source, matchLoc, new Point(matchLoc.x + template.width(), matchLoc.y + template.height()), new Scalar(0, 255, 0));
        //将结果输出到对应位置
        Imgcodecs.imwrite(targetImg, source);
    }

    /**
     * 基于特征点的SURF匹配算法——模板图片发生了旋转、缩放之后仍然可以精准匹配
     * @param templateFilePath 模板图url
     * @param originalFilePath 原图url
     */
    public static Integer getMatchImage(String templateFilePath, String originalFilePath) {
        Mat templateImage = Imgcodecs.imread(templateFilePath, Imgcodecs.CV_LOAD_IMAGE_COLOR);
        Mat originalImage = Imgcodecs.imread(originalFilePath, Imgcodecs.CV_LOAD_IMAGE_COLOR);
        MatOfKeyPoint templateKeyPoints = new MatOfKeyPoint();
        //指定特征点算法SURF
        FeatureDetector featureDetector = FeatureDetector.create(FeatureDetector.SURF);
        //获取模板图的特征点
        featureDetector.detect(templateImage, templateKeyPoints);
        //提取模板图的特征点
        MatOfKeyPoint templateDescriptors = new MatOfKeyPoint();
        DescriptorExtractor descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
        System.out.println("提取模板图的特征点");
        descriptorExtractor.compute(templateImage, templateKeyPoints, templateDescriptors);

        //显示模板图的特征点图片
        Mat outputImage = new Mat(templateImage.rows(), templateImage.cols(), Imgcodecs.CV_LOAD_IMAGE_COLOR);
        System.out.println("在图片上显示提取的特征点");
        Features2d.drawKeypoints(templateImage, templateKeyPoints, outputImage, new Scalar(255, 0, 0), 0);

        //获取原图的特征点
        MatOfKeyPoint originalKeyPoints = new MatOfKeyPoint();
        MatOfKeyPoint originalDescriptors = new MatOfKeyPoint();
        featureDetector.detect(originalImage, originalKeyPoints);
        System.out.println("提取原图的特征点");
        descriptorExtractor.compute(originalImage, originalKeyPoints, originalDescriptors);

        List<MatOfDMatch> matches = new LinkedList();
        DescriptorMatcher descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        System.out.println("寻找最佳匹配");
        /**
         * knnMatch方法的作用就是在给定特征描述集合中寻找最佳匹配
         * 使用KNN-matching算法，令K=2，则每个match得到两个最接近的descriptor，然后计算最接近距离和次接近距离之间的比值，当比值大于既定值时，才作为最终match。
         */
        try {
            descriptorMatcher.knnMatch(templateDescriptors, originalDescriptors, matches, 2);
        } catch (Exception e) {
//            e.printStackTrace();
//            e.getMessage();
            System.out.println("特征点匹配出现异常");
            return 0;
        }

        System.out.println("计算匹配结果");
        LinkedList<DMatch> goodMatchesList = distanceMatch(matches);
        //获取图片比对特征点数
        Integer matchesPointCount = goodMatchesList.size();
        //图片相似度识别
        smartImage(matchesPointCount, templateKeyPoints, originalKeyPoints, goodMatchesList, templateImage, originalImage);
        //保存模板特征点采集后的图片
        Imgcodecs.imwrite("D:/usr/local/picture/matchImg.jpg", outputImage);
        return matchesPointCount;
    }

    public static Integer getMatchImage(Mat templateImage, Mat originalImage) {
        MatOfKeyPoint templateKeyPoints = new MatOfKeyPoint();
        //指定特征点算法SURF
        FeatureDetector featureDetector = FeatureDetector.create(FeatureDetector.SURF);
        //获取模板图的特征点
        featureDetector.detect(templateImage, templateKeyPoints);
        //提取模板图的特征点
        MatOfKeyPoint templateDescriptors = new MatOfKeyPoint();
        DescriptorExtractor descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.SURF);
        System.out.println("提取模板图的特征点");
        descriptorExtractor.compute(templateImage, templateKeyPoints, templateDescriptors);

        //显示模板图的特征点图片
        Mat outputImage = new Mat(templateImage.rows(), templateImage.cols(), Imgcodecs.CV_LOAD_IMAGE_COLOR);
        System.out.println("在图片上显示提取的特征点");
        Features2d.drawKeypoints(templateImage, templateKeyPoints, outputImage, new Scalar(255, 0, 0), 0);

        //获取原图的特征点
        MatOfKeyPoint originalKeyPoints = new MatOfKeyPoint();
        MatOfKeyPoint originalDescriptors = new MatOfKeyPoint();
        featureDetector.detect(originalImage, originalKeyPoints);
        System.out.println("提取原图的特征点");
        descriptorExtractor.compute(originalImage, originalKeyPoints, originalDescriptors);

        List<MatOfDMatch> matches = new LinkedList();
        DescriptorMatcher descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        System.out.println("寻找最佳匹配");
        /**
         * knnMatch方法的作用就是在给定特征描述集合中寻找最佳匹配
         * 使用KNN-matching算法，令K=2，则每个match得到两个最接近的descriptor，然后计算最接近距离和次接近距离之间的比值，当比值大于既定值时，才作为最终match。
         */
        try {
            descriptorMatcher.knnMatch(templateDescriptors, originalDescriptors, matches, 2);
        } catch (Exception e) {
//            e.printStackTrace();
//            e.getMessage();
            System.out.println("特征点匹配出现异常");
            return 0;
        }
        System.out.println("计算匹配结果");
        LinkedList<DMatch> goodMatchesList = distanceMatch(matches);
        //获取图片比对特征点数
        Integer matchesPointCount = goodMatchesList.size();
        //图片相似度识别
        smartImage(matchesPointCount, templateKeyPoints, originalKeyPoints, goodMatchesList, templateImage, originalImage);
        //保存模板特征点采集后的图片
        Imgcodecs.imwrite("D:/usr/local/dnn/matchImg.jpg", outputImage);
        return matchesPointCount;
    }

    public static LinkedList<DMatch> distanceMatch(List<MatOfDMatch> matches){
        LinkedList<DMatch> goodMatchesList = new LinkedList();
        //对匹配结果进行筛选，依据distance进行筛选
        matches.forEach(match -> {
            DMatch[] dmatcharray = match.toArray();
            DMatch m1 = dmatcharray[0];
            DMatch m2 = dmatcharray[1];

            if (m1.distance <= m2.distance * nndrRatio) {
                goodMatchesList.addLast(m1);
            }
        });
        return goodMatchesList;
    }

    /**
     * 图片特征点深度计算方法
     * @param pointCount
     * @param templateKeyPoints
     * @param originalKeyPoints
     * @param goodMatchesList
     * @param templateImage
     * @param originalImage
     */
    public static void smartImage(Integer pointCount, MatOfKeyPoint templateKeyPoints, MatOfKeyPoint originalKeyPoints,
                                  LinkedList<DMatch> goodMatchesList, Mat templateImage, Mat originalImage) {
        if(null != pointCount){
            //当匹配后的特征点大于等于 4 个，则认为模板图在原图中，该值可以自行调整(官方文档要求至少4个，小于4会抛异常)
            if (pointCount >= 4) {

                List<KeyPoint> templateKeyPointList = templateKeyPoints.toList();
                List<KeyPoint> originalKeyPointList = originalKeyPoints.toList();
                LinkedList<Point> objectPoints = new LinkedList();
                LinkedList<Point> scenePoints = new LinkedList();
                goodMatchesList.forEach(goodMatch -> {
                    objectPoints.addLast(templateKeyPointList.get(goodMatch.queryIdx).pt);
                    scenePoints.addLast(originalKeyPointList.get(goodMatch.trainIdx).pt);
                });
                MatOfPoint2f objMatOfPoint2f = new MatOfPoint2f();
                objMatOfPoint2f.fromList(objectPoints);
                MatOfPoint2f scnMatOfPoint2f = new MatOfPoint2f();
                scnMatOfPoint2f.fromList(scenePoints);
                //使用 findHomography 寻找匹配上的关键点的变换
                Mat homography = Calib3d.findHomography(objMatOfPoint2f, scnMatOfPoint2f, Calib3d.RANSAC, 3);

                /**
                 * 透视变换(Perspective Transformation)是将图片投影到一个新的视平面(Viewing Plane)，也称作投影映射(Projective Mapping)。
                 */
                Mat templateCorners = new Mat(4, 1, CvType.CV_32FC2);
                Mat templateTransformResult = new Mat(4, 1, CvType.CV_32FC2);
                templateCorners.put(0, 0, new double[]{0, 0});
                templateCorners.put(1, 0, new double[]{templateImage.cols(), 0});
                templateCorners.put(2, 0, new double[]{templateImage.cols(), templateImage.rows()});
                templateCorners.put(3, 0, new double[]{0, templateImage.rows()});
                //使用 perspectiveTransform 将模板图进行透视变以矫正图象得到标准图片
                Core.perspectiveTransform(templateCorners, templateTransformResult, homography);

                //矩形四个顶点
                double[] pointA = templateTransformResult.get(0, 0);
                double[] pointB = templateTransformResult.get(1, 0);
                double[] pointC = templateTransformResult.get(2, 0);
                double[] pointD = templateTransformResult.get(3, 0);

                //指定取得数组子集的范围
                int rowStart = (int) Math.abs(pointA[1]);
                int rowEnd = (int) Math.abs(pointC[1]);
                int colStart = (int) Math.abs(pointD[0]);
                int colEnd = (int) Math.abs(pointB[0]);
                //准备临时变量
                int tempRowStart = rowStart;
                int tempColStart = colStart;
                if(rowStart >= rowEnd){
                    rowStart = rowEnd;
                    Integer height = originalImage.rows();
                    if(height > tempRowStart){
                        rowEnd = height;
                    }else{
                        rowEnd = tempRowStart;
                    }
                }
                if(colStart >= colEnd){
                    colStart = colEnd;
                    Integer width = originalImage.cols();
                    if(width > tempColStart){
                        colEnd = width;
                    }else{
                        colEnd = tempColStart;
                    }
                }
                System.out.println("rowStart="+rowStart+";rowEnd="+rowEnd+";colStart="+colStart+";colEnd="+colEnd);
                System.out.println("模板图和原图匹配成功！");
                //rowStart=-2336;rowEnd=3133;colStart=-1153;colEnd=2536(error)
                //rowStart=520;rowEnd=904;colStart=396;colEnd=709
                try {
                    Mat subMat = originalImage.submat(rowStart, rowEnd, colStart, colEnd);
                    Imgcodecs.imwrite("D:/usr/local/dnn/old.jpg", subMat);
                } catch (Exception e) {
                    System.out.println("DNN图片计算异常");
                    return;
                }
                //将匹配的图像用用四条线框出来
                line(originalImage, new Point(pointA), new Point(pointB), new Scalar(0, 255, 0), 4);//上 A->B
                line(originalImage, new Point(pointB), new Point(pointC), new Scalar(0, 255, 0), 4);//右 B->C
                line(originalImage, new Point(pointC), new Point(pointD), new Scalar(0, 255, 0), 4);//下 C->D
                line(originalImage, new Point(pointD), new Point(pointA), new Scalar(0, 255, 0), 4);//左 D->A

                MatOfDMatch goodMatches = new MatOfDMatch();
                goodMatches.fromList(goodMatchesList);
                Mat matchOutput = new Mat(originalImage.rows() * 2, originalImage.cols() * 2, Imgcodecs.CV_LOAD_IMAGE_COLOR);
                Features2d.drawMatches(templateImage, templateKeyPoints, originalImage, originalKeyPoints, goodMatches, matchOutput, new Scalar(0, 255, 0), new Scalar(255, 0, 0), new MatOfByte(), 2);

                Imgcodecs.imwrite("D:/usr/local/dnn/surf.jpg", matchOutput);
                Imgcodecs.imwrite("D:/usr/local/dnn/otm.jpg", originalImage);
            } else {
                System.out.println("模板图和原图不匹配！");
            }
        }
    }

    /**
     * 非递归方式调用文件夹下的所有文件
     * @param path
     */
    public static void listFolder(String path) {
        int fileNum = 0, folderNum = 0;
        File file = new File(path);
        if (file.exists()) {
            LinkedList<File> list = new LinkedList<File>();
            File[] files = file.listFiles();
            for (File file2 : files) {
                if (file2.isDirectory()) {
                    System.out.println("文件夹:" + file2.getAbsolutePath());
                    list.add(file2);
                    folderNum++;
                } else {
                    System.out.println("文件:" + file2.getAbsolutePath());
                    fileNum++;
                }
            }
            File temp_file;
            while (!list.isEmpty()) {
                temp_file = list.removeFirst();
                files = temp_file.listFiles();
                for (File file2 : files) {
                    if (file2.isDirectory()) {
                        System.out.println("文件夹:" + file2.getAbsolutePath());
                        list.add(file2);
                        folderNum++;
                    } else {
                        System.out.println("文件:" + file2.getAbsolutePath());
                        fileNum++;
                    }
                }
            }
        } else {
            System.out.println("文件不存在!");
        }
        System.out.println("文件夹共有:" + folderNum + ",文件共有:" + fileNum);
    }

    /**
     * 递归方式调用文件夹下的所有文件
     * @param path
     */
    public static void recursiveFolder(String path) {
        File file = new File(path);
        if (file.exists()) {
            File[] files = file.listFiles();
            if (null == files || files.length == 0) {
                System.out.println("文件夹是空的!");
                return;
            } else {
                for (File file2 : files) {
                    if (file2.isDirectory()) {
                        System.out.println("文件夹:" + file2.getAbsolutePath());
                        recursiveFolder(file2.getAbsolutePath());
                    } else {
                        System.out.println("文件:" + file2.getAbsolutePath());
                    }
                }
            }
        } else {
            System.out.println("文件不存在!");
        }
    }

    /**
     * 获取文件夹中所有图片的对比结果
     * @param grayImg 模板图片
     * @return
     */
    public static Map<String, Integer> getMatchImageMap(String foldr, Mat grayImg) {
        Map<String, Integer> matchResult = new HashMap<>();
        File dir = new File(folderUrl);
        File[] files = dir.listFiles(); // 该文件目录下文件全部放入数组
        if (files != null) {
            for (int i = 0; i < files.length; i++) {
                String fileName = files[i].getName();
                if (files[i].isDirectory()) { // 判断是文件还是文件夹
                    getMatchImageMap(files[i].getAbsolutePath(), grayImg); // 获取文件绝对路径
                } else if (fileName.endsWith("jpg") || fileName.endsWith("png")) { // 判断文件名是否以.jpg结尾
                    String originalImageUrl = files[i].getAbsolutePath();
                    Mat originalImage = Imgcodecs.imread(originalImageUrl);
                    //当前图片转换成灰度图片
                    Mat originalGrayImage = new Mat(originalImage.rows(),originalImage.cols(),originalImage.type());
                    Imgproc.cvtColor(originalImage, originalGrayImage, Imgproc.COLOR_RGB2GRAY);
                    System.out.println(originalImageUrl);
                    Integer matchPoints = getMatchImage(grayImg, originalGrayImage);
                    matchResult.put(originalImageUrl, matchPoints);
                } else {
                    continue;
                }
            }

        }
        return matchResult;
    }

    /**
     * map集合-根据value获取key
     * @param map
     * @param value
     * @return
     */
    public static List<Object> getKey(Map map, Object value){
        List<Object> keyList = new ArrayList<>();
        for(Object key: map.keySet()){
            if(map.get(key).equals(value)){
                keyList.add(key);
            }
        }
        return keyList;
    }

    /**
     * 图片灰度处理
     * @param mrgb
     * @return
     */
    public static Mat rgbToGray(Mat mrgb){
        Mat mgray=new Mat(mrgb.rows(),mrgb.cols(),CvType.CV_8UC1);//一样大小的灰度图
        for(int i=0;i<mrgb.rows();i++){
            for(int j=0;j<mrgb.cols();j++){
                double[] value=mrgb.get(i, j);//BGR顺序
                double valueGray=0.114*value[0]+0.587*value[1]+0.299*value[2];//转换公式
                mgray.put(i, j, valueGray);
            }
        }
        return mgray;
    }

}
