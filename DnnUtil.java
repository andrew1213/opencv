package com.plansolve.farm.util;

import com.plansolve.farm.model.OpencvInstance;
import org.apache.commons.lang3.StringUtils;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.ANN_MLP;
import org.opencv.ml.Boost;
import org.opencv.ml.DTrees;
import org.opencv.ml.KNearest;
import org.opencv.ml.LogisticRegression;
import org.opencv.ml.Ml;
import org.opencv.ml.NormalBayesClassifier;
import org.opencv.ml.RTrees;
import org.opencv.ml.SVM;
import org.opencv.ml.SVMSGD;
import org.opencv.ml.TrainData;

/**
 * @Author: Andrew
 * @Date: 2019/2/13
 * @Description: https://blog.csdn.net/marooon/article/details/80265247（参考文档1）
 *               https://blog.csdn.net/wishchin/article/details/78778741（参考文档2）
 */
public class DnnUtil {

    private static final String DNN_PREDICT_RESULT = "failed";

    private static final String DAOQUBING_SMART_ARITHMETIC = "D:/usr/local/arithmetic/daoqubing.xml";

    private static final String DAOWENBING_SMART_ARITHMETIC = "D:/usr/local/arithmetic/daowenbing.xml";

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    /**
     * @author idmin
     * 样本的数量
     */
    private  static final int DAOQUBING_SAMPLE_NUMBER=54;

    private  static final int DAOWENBING_SAMPLE_NUMBER=58;

    public static void main(String[] args) {
        /**
         * 分类器训练代码
         */
        //用于存放所有样本矩阵
//        Mat trainingDataMat0 = null;
//
//        //标记：正样本用 1 表示，负样本用 0 表示。
//        //图片命名：
//        //正样本： 0.jpg  1.jpg  2.jpg  3.jpg  4.jpg
//        //负样本：5.jpg  6.jpg  7.jpg   ...   17.jpg
//        int labels[]  = {
//                1,1,1,1,1,1,1,1,1,1,
//                0,0,0,0,0,0,0,0,0,0,
//                0,0,0,0,0,0,0,0,0,0,
//                0,0,0,0,0,0,0,0,0};
//
//        //存放标记的Mat,每个图片都要给一个标记。SAMPLE_NUMBER 是自己定义的图片数量
//        Mat labelsMat = new Mat(SAMPLE_NUMBER,1,CvType.CV_32SC1);
//        labelsMat.put(0, 0, labels);
//
//        //这里的意思是，trainingDataMat 存放18张图片的矩阵，trainingDataMat 的每一行存放一张图片的矩阵。
//        for(int i = 0;i<SAMPLE_NUMBER;i++) {
//            String path = "D:\\photos\\daoqubing\\" + i + ".jpg" ;
//            Mat src = Imgcodecs.imread(path);
//            //将训练图片缩放至指定尺寸
//            Imgproc.resize(src, src, new Size(200,200), 0, 0, Imgproc.INTER_AREA);
//            //创建一个行数为SAMPLE_NUMBER(正负样本总数量为SAMPLE_NUMBER),列数为 rows*cols 的矩阵
//            if(trainingDataMat0 == null) {
//                trainingDataMat0 = new Mat(SAMPLE_NUMBER, src.rows()*src.cols(),CvType.CV_32FC1);// CV_32FC1 是规定的训练用的图片格式。
//            }
//
//            //转成灰度图并检测边缘
//            //这里是为了过滤不需要的特征，减少训练时间。实际处理按情况论。
//            Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
//            Mat dst = new Mat(src.rows(),src.cols(),src.type());//此时的 dst 是8u1c。
//            Imgproc.Canny(src, dst, 130, 250);
//
//            //转成数组再添加。
//            //失败案例:这里我试图用 get(row,col,data)方法获取数组，但是结果和这个结果不一样，原因未知。
//            float[] arr =new float[dst.rows()*dst.cols()];
//            int l=0;
//            for (int j=0;j<dst.rows();j++){
//                for(int k=0;k<dst.cols();k++) {
//                    double[] a=dst.get(j, k);
//                    arr[l]=(float)a[0];
//                    l++;
//                }
//            }
//            trainingDataMat0.put(i, 0, arr);
//        }
//        //每次训练的结果得到的 xml 文件都不一样，原因未知。猜测是我的样本数太太太太小
//        MySvm(trainingDataMat0, labelsMat, "D:\\photos\\daoqubing\\f.xml");


        //图片预测
//        Mat src = Imgcodecs.imread("D:\\photos\\daoqubing\\t5.jpg");//图片大小要和样本一致
//        Imgproc.resize(src, src, new Size(200,200), 0, 0, Imgproc.INTER_AREA);
//        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
//        Mat dst = new Mat();
//        Imgproc.Canny(src, dst, 40, 200);
//        test(dst);

        /**
         * 测试其它算法
         */
        //准备样本数据
        Mat trainingDataMat = prepaireTrainData(DAOWENBING_SAMPLE_NUMBER, OpencvInstance.DAOWENBING_FOLDER);
        //存放所有样本标记的Mat,每个图片都要给一个标记。SAMPLE_NUMBER 是自己定义的图片数量
        Mat labelsMat = getSampleLabelsMat(DAOWENBING_SAMPLE_NUMBER, OpencvInstance.DAOWENBING_LABELS);
        //训练分类器
//        DtreesTrain(trainingDataMat, labelsMat, DAOWENBING_SMART_ARITHMETIC);

        //准备测试样本
        Mat testMat = getPredictData("D:\\photos\\daoqubing\\t5.jpg");
        //样本预测
        String result = MyDtrees(testMat, DAOWENBING_SMART_ARITHMETIC);


//        if(StringUtils.isNotBlank(result)){
//            if (!result.equals(DNN_PREDICT_RESULT)){
//
//            }
//        }

        /**
         * 其它分类算法
         */
//        MyAnn(trainingDataMat, labelsMat1, sampleMat);
//        MyBoost(trainingDataMat, labelsMat, sampleMat);
//        MyNormalBayes(trainingDataMat, labelsMat, sampleMat);
//        MyRTrees(trainingDataMat, labelsMat, sampleMat);
//        MySvm(trainingDataMat, labelsMat, sampleMat);
//        MySvmsgd(trainingDataMat, labelsMat, sampleMat);
//        MyKnn(trainingDataMat, labelsMat3, sampleMat);
//        MyLogisticRegression(trainingDataMat, labelsMat3, sampleMat);

    }

    /**
     * 根据标签数量和指定路径准备分类器训练用的mat
     * @param sampleNumber
     * @param folder
     * @return
     */
    public static Mat prepaireTrainData(Integer sampleNumber, String folder){
        Mat trainingDataMat = null;
        //这里的意思是，trainingDataMat 存放18张图片的矩阵，trainingDataMat 的每一行存放一张图片的矩阵。
        for(int i = 0;i<sampleNumber;i++) {
            String path = folder + i + ".jpg" ;
            Mat src = Imgcodecs.imread(path);
            //将训练图片缩放至指定尺寸
            Imgproc.resize(src, src, new Size(200,200), 0, 0, Imgproc.INTER_AREA);
            //创建一个行数为SAMPLE_NUMBER(正负样本总数量为SAMPLE_NUMBER),列数为 rows*cols 的矩阵
            if(trainingDataMat == null) {
                trainingDataMat = new Mat(sampleNumber, src.rows()*src.cols(),CvType.CV_32FC1);// CV_32FC1 是规定的训练用的图片格式。
            }
            //转成灰度图并检测边缘
            //这里是为了过滤不需要的特征，减少训练时间。实际处理按情况论。
            Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
            Mat dst = new Mat(src.rows(),src.cols(),CvType.CV_32FC1);//此时的 dst 是8u1c。
            Imgproc.Canny(src, dst, 130, 250);
            //获取所有图片的矩阵数据-转成数组再添加
            float[] arr = getFloatArr(dst);
            trainingDataMat.put(i, 0, arr);
        }
        return trainingDataMat;
    }

    /**
     * 根据指定的标签数量和标签数组信息来生成对应的Mat
     * @param sampleNumber
     * @param sampleLabels
     * @return
     */
    public static Mat getSampleLabelsMat(Integer sampleNumber, int[] sampleLabels){
        //存放标记的Mat,每个图片都要给一个标记。SAMPLE_NUMBER 是自己定义的图片数量
        Mat labelsMat = new Mat(sampleNumber,1,CvType.CV_32SC1);
        labelsMat.put(0, 0, sampleLabels);
        return labelsMat;
    }

    /**
     * 根据url路径获取数据信息并保存到生成的Mat中
     * @param testDataUrl
     * @return
     */
    public static Mat getPredictData(String testDataUrl){
        //测试数据
        Mat src = Imgcodecs.imread(testDataUrl);//图片大小要和样本一致
        Mat dst = new Mat(1,1, CvType.CV_32FC1);
        Imgproc.resize(src, src, new Size(200,200), 0, 0, Imgproc.INTER_AREA);
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
        Imgproc.Canny(src, dst, 40, 200);
        //获取所有图片的矩阵数据-转成数组再添加
        float[] arr = getFloatArr(dst);
        Mat sampleMat = new Mat(1, dst.rows()*dst.cols(),CvType.CV_32FC1);
        sampleMat.put(0, 0, arr);
        return sampleMat;
    }

    /**
     * 将目标Mat的数据信息放到对应大小的float数组
     * @param dstMat
     * @return
     */
    public static float[] getFloatArr(Mat dstMat){
        float[] arr =new float[dstMat.rows()*dstMat.cols()];
        int l=0;
        for (int j=0;j<dstMat.rows();j++){
            for(int k=0;k<dstMat.cols();k++) {
                double[] a=dstMat.get(j, k);
                arr[l]=(float)a[0];
                l++;
            }
        }
        return arr;
    }

    /**
     * SVM 支持向量机
     * @param trainingDataMat 存放样本的矩阵
     * @param labelsMat 存放标识的矩阵
     * @param savePath 保存路径 。例如：d:/svm.xml
     */
    public static void MySvm(Mat trainingDataMat, Mat labelsMat, String savePath) {
        SVM svm = SVM.create();
        // 配置SVM训练器参数
        TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 1000, 0);
        svm.setTermCriteria(criteria);// 指定
        svm.setKernel(SVM.LINEAR);// 使用预先定义的内核初始化
        svm.setType(SVM.C_SVC); // SVM的类型,默认是：SVM.C_SVC
        svm.setGamma(0.5);// 核函数的参数
        svm.setNu(0.5);// SVM优化问题参数
        svm.setC(1);// SVM优化问题的参数C

        TrainData td = TrainData.create(trainingDataMat, Ml.ROW_SAMPLE, labelsMat);// 类封装的训练数据
        boolean success = svm.train(td.getSamples(), Ml.ROW_SAMPLE, td.getResponses());// 训练统计模型
        System.out.println("Svm training result: " + success);
        svm.save(savePath);// 保存模型
    }

    /**
     * 图片预测
     * @param src
     */
    public static void test(Mat src) {
        SVM svm = SVM.load("D:\\photos\\daoqubing\\f.xml");//加载训练得到的 xml

        Mat samples = new Mat(1,src.cols()*src.rows(),CvType.CV_32FC1);

        //转换 src 图像的 cvtype
        //失败案例：我试图用 src.convertTo(src, CvType.CV_32FC1); 转换，但是失败了，原因未知。猜测: 内部的数据类型没有转换？
        float[] dataArr = new float[src.cols()*src.rows()];
        for(int i =0,f = 0 ;i<src.rows();i++) {
            for(int j = 0;j<src.cols();j++) {
                float pixel = (float)src.get(i, j)[0];
                dataArr[f] = pixel;
                f++;
            }
        }
        samples.put(0, 0, dataArr);

        //预测用的方法，返回定义的标识。
//      int labels[]  = {9,9,9,9,
//                 1,1,1,1,1,1,1,
//                 1,1,1,1,1,1,1};
//      如果训练时使用这个标识，那么符合的图像会返回9.0
        float flag = svm.predict(samples);

        System.out.println("预测结果："+flag);
        if(flag == 0) {
            System.out.println("目标不符合");
        }
        if(flag == 1) {
            System.out.println("目标符合");
        }
    }

    /**
     * 决策树预测方法
     * @param testData 测试数据
     * @param modelSavingUrl 指定算法地址
     * @return
     */
    public static String MyDtrees(Mat testData, String modelSavingUrl) {
        DTrees dtree = DTrees.load(modelSavingUrl);
        float response = dtree.predict(testData);
        System.out.println("预测结果："+response);
        if(response == 1) {
            String result = modelSavingUrl.substring(modelSavingUrl.lastIndexOf("/")+1, modelSavingUrl.indexOf("."));
            System.out.println("病虫害名称是："+result);
            return result;
        }else{
            return DNN_PREDICT_RESULT;
        }
    }

    /**
     * 决策树训练方法
     * @param trainingData 分类样本数据
     * @param labels 分类样本标签
     * @param modelSavingUrl 分类器文本存储路径
     * @return
     */
    public static String DtreesTrain(Mat trainingData, Mat labels, String modelSavingUrl) {
        DTrees dtree = DTrees.create(); // 创建分类器
        dtree.setMaxDepth(8); // 设置最大深度
        dtree.setMinSampleCount(2);
        dtree.setUseSurrogates(false);
        dtree.setCVFolds(0); // 交叉验证
        dtree.setUse1SERule(false);
        dtree.setTruncatePrunedTree(false);

        TrainData td = TrainData.create(trainingData, Ml.ROW_SAMPLE, labels);
        boolean success = dtree.train(td.getSamples(), Ml.ROW_SAMPLE, td.getResponses());
        System.out.println("Dtrees training result: " + success);
        //训练成功，将分类器文本存入指定地址
        if(success){
            dtree.save(modelSavingUrl);//存储模型
            return modelSavingUrl;
        }else{
            return DNN_PREDICT_RESULT;
        }
    }

    // 人工神经网络
    public static Mat MyAnn(Mat trainingData, Mat labels, Mat testData) {
        // train data using aNN
        TrainData td = TrainData.create(trainingData, Ml.ROW_SAMPLE, labels);
        Mat layerSizes = new Mat(1, 4, CvType.CV_32FC1);
        // 含有两个隐含层的网络结构，输入、输出层各两个节点，每个隐含层含两个节点
        layerSizes.put(0, 0, new float[] { 2, 2, 2, 2 });
        ANN_MLP ann = ANN_MLP.create();
        ann.setLayerSizes(layerSizes);
        ann.setTrainMethod(ANN_MLP.BACKPROP);
        ann.setBackpropWeightScale(0.1);
        ann.setBackpropMomentumScale(0.1);
        ann.setActivationFunction(ANN_MLP.SIGMOID_SYM, 1, 1);
        ann.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER + TermCriteria.EPS, 300, 0.0));
        boolean success = ann.train(td.getSamples(), Ml.ROW_SAMPLE, td.getResponses());
        System.out.println("Ann training result: " + success);
        // ann.save("D:/bp.xml");//存储模型
        // ann.load("D:/bp.xml");//读取模型

        // 测试数据
        Mat responseMat = new Mat();
        ann.predict(testData, responseMat, 0);
        System.out.println("Ann responseMat:\n" + responseMat.dump());
        for (int i = 0; i < responseMat.size().height; i++) {
            if (responseMat.get(i, 0)[0] + responseMat.get(i, i)[0] >= 1)
                System.out.println("Girl\n");
            if (responseMat.get(i, 0)[0] + responseMat.get(i, i)[0] < 1)
                System.out.println("Boy\n");
        }
        return responseMat;
    }

    // Boost
    public static Mat MyBoost(Mat trainingData, Mat labels, Mat testData) {
        Boost boost = Boost.create();
        // boost.setBoostType(Boost.DISCRETE);
        boost.setBoostType(Boost.GENTLE);
        boost.setWeakCount(2);
        boost.setWeightTrimRate(0.95);
        boost.setMaxDepth(2);
        boost.setUseSurrogates(false);
        boost.setPriors(new Mat());

        TrainData td = TrainData.create(trainingData, 0, labels);
        boolean success = boost.train(td.getSamples(), 0, td.getResponses());
        System.out.println("Boost training result: " + success);
        boost.save("D:/boost.xml");//存储模型

        Mat responseMat = new Mat();
        float response = boost.predict(testData, responseMat, 0);
        System.out.println("预测结果："+response);
        if(response == 0) {
            System.out.println("目标不符合");
        }
        if(response == 1) {
            System.out.println("目标符合");
        }
        return responseMat;
    }

    // K最邻近
    public static Mat MyKnn(Mat trainingData, Mat labels, Mat testData) {
        final int K = 2;
        TrainData td = TrainData.create(trainingData, Ml.ROW_SAMPLE, labels);
        KNearest knn = KNearest.create();
        boolean success = knn.train(trainingData, Ml.ROW_SAMPLE, labels);
        System.out.println("Knn training result: " + success);
        // knn.save("D:/bp.xml");//存储模型

        // find the nearest neighbours of test data
        Mat results = new Mat();
        Mat neighborResponses = new Mat();
        Mat dists = new Mat();
        float response = knn.findNearest(testData, K, results, neighborResponses, dists);
        System.out.println("results:\n" + results.dump());
        System.out.println("Knn neighborResponses:\n" + neighborResponses.dump());
        System.out.println("dists:\n" + dists.dump());
        System.out.println("预测结果："+response);
        if(response == 0) {
            System.out.println("目标不符合");
        }
        if(response == 1) {
            System.out.println("目标符合");
        }

        return results;
    }

    // 逻辑回归
    public static Mat MyLogisticRegression(Mat trainingData, Mat labels, Mat testData) {
        LogisticRegression lr = LogisticRegression.create();

        TrainData td = TrainData.create(trainingData, Ml.ROW_SAMPLE, labels);
        boolean success = lr.train(td.getSamples(), Ml.ROW_SAMPLE, td.getResponses());
        System.out.println("LogisticRegression training result: " + success);
        // lr.save("D:/bp.xml");//存储模型

        Mat responseMat = new Mat();
        float response = lr.predict(testData, responseMat, 0);
        System.out.println("LogisticRegression responseMat:\n" + responseMat.dump());
        for (int i = 0; i < responseMat.height(); i++) {
            if (responseMat.get(i, 0)[0] == 0)
                System.out.println("Boy\n");
            if (responseMat.get(i, 0)[0] == 1)
                System.out.println("Girl\n");
        }
        return responseMat;
    }

    // 贝叶斯
    public static Mat MyNormalBayes(Mat trainingData, Mat labels, Mat testData) {
        NormalBayesClassifier nb = NormalBayesClassifier.create();

        TrainData td = TrainData.create(trainingData, Ml.ROW_SAMPLE, labels);
        boolean success = nb.train(td.getSamples(), Ml.ROW_SAMPLE, td.getResponses());
        System.out.println("NormalBayes training result: " + success);
        nb.save("D:/bys.xml");//存储模型

        Mat responseMat = new Mat();
        float response = nb.predict(testData, responseMat, 0);
        System.out.println("预测结果："+response);
        if(response == 0) {
            System.out.println("目标不符合");
        }
        if(response == 1) {
            System.out.println("目标符合");
        }
        return responseMat;
    }

    // 随机森林
    public static Mat MyRTrees(Mat trainingData, Mat labels, Mat testData) {
        RTrees rtrees = RTrees.create();
        rtrees.setMaxDepth(4);
        rtrees.setMinSampleCount(2);
        rtrees.setRegressionAccuracy(0.f);
        rtrees.setUseSurrogates(false);
        rtrees.setMaxCategories(16);
        rtrees.setPriors(new Mat());
        rtrees.setCalculateVarImportance(false);
        rtrees.setActiveVarCount(1);
        rtrees.setTermCriteria(new TermCriteria(TermCriteria.MAX_ITER, 5, 0));
        TrainData tData = TrainData.create(trainingData, Ml.ROW_SAMPLE, labels);
        boolean success = rtrees.train(tData.getSamples(), Ml.ROW_SAMPLE, tData.getResponses());
        System.out.println("Rtrees training result: " + success);
        rtrees.save("D:\\sjsl.xml");//存储模型

        Mat responseMat = new Mat();
        float response = rtrees.predict(testData, responseMat, 0);
        System.out.println("预测结果："+response);
        if(response == 0) {
            System.out.println("目标不符合");
        }
        if(response == 1) {
            System.out.println("目标符合");
        }
        return responseMat;
    }

    // 支持向量机
    public static Mat MySvm(Mat trainingData, Mat labels, Mat testData) {
        SVM svm = SVM.create();
        svm.setKernel(SVM.LINEAR);
        svm.setType(SVM.C_SVC);
        TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 1000, 0);
        svm.setTermCriteria(criteria);
        svm.setGamma(0.5);
        svm.setNu(0.5);
        svm.setC(1);

        TrainData td = TrainData.create(trainingData, Ml.ROW_SAMPLE, labels);
        boolean success = svm.train(td.getSamples(), Ml.ROW_SAMPLE, td.getResponses());
        System.out.println("Svm training result: " + success);
        // svm.save("D:/bp.xml");//存储模型
        // svm.load("D:/bp.xml");//读取模型

        Mat responseMat = new Mat();
        float response = svm.predict(testData, responseMat, 0);
        System.out.println("预测结果："+response);
        if(response == 0) {
            System.out.println("目标不符合");
        }
        if(response == 1) {
            System.out.println("目标符合");
        }
        return responseMat;
    }

    // SGD支持向量机
    public static Mat MySvmsgd(Mat trainingData, Mat labels, Mat testData) {
        SVMSGD Svmsgd = SVMSGD.create();
        TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 1000, 0);
        Svmsgd.setTermCriteria(criteria);
        Svmsgd.setInitialStepSize(2);
        Svmsgd.setSvmsgdType(SVMSGD.SGD);
        Svmsgd.setMarginRegularization(0.5f);
        boolean success = Svmsgd.train(trainingData, Ml.ROW_SAMPLE, labels);
        System.out.println("SVMSGD training result: " + success);
        // svm.save("D:/bp.xml");//存储模型
        // svm.load("D:/bp.xml");//读取模型

        Mat responseMat = new Mat();
        float response = Svmsgd.predict(testData, responseMat, 0);
        System.out.println("预测结果："+response);
        if(response == 0) {
            System.out.println("目标不符合");
        }
        if(response == 1) {
            System.out.println("目标符合");
        }
        return responseMat;
    }


}





