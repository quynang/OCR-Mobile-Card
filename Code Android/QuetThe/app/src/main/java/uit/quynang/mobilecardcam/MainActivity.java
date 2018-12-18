package uit.quynang.mobilecardcam;

import android.Manifest;
import android.app.Dialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.hardware.Camera;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.support.design.widget.FloatingActionButton;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.KeyEvent;
import android.view.LayoutInflater;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.SVM;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;


public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static String TAG = "MainActivity";
    JavaCameraView javaCameraView;
    Mat mRgba;
    private static final int  MY_PERMISSIONS_REQUEST_CAMERA = 101;
    private static final int   MY_PERMISSIONS_REQUEST_READ_EXTERNAL_STORAGE = 102;
    private static final int  MY_PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE = 103;
    private static final int  MY_PERMISSIONS_REQUEST_CALL_PHONE = 104;
    private Context mContext;
    private int screenWidth;
    private int screenHeight;
    private String trainningFile;
    private Timer detectTimer;
    private int x, y, m;
    private int width,height,per;
    Button btnHuongDan;
    private Rect boudingRect;
    private Rect boudingRect2;
    private SVM svm;
    Mat imgCut;
    private String cardNum = "";
    BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case BaseLoaderCallback.SUCCESS: {
                    javaCameraView.enableView();
                    break;
                }
                default: {
                    super.onManagerConnected(status);
                }


            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA},
                    MY_PERMISSIONS_REQUEST_CAMERA);

        }

        if (Build.VERSION.SDK_INT >= 23) {
            if (ContextCompat.checkSelfPermission(this,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                        MY_PERMISSIONS_REQUEST_WRITE_EXTERNAL_STORAGE);

            }

            if (ContextCompat.checkSelfPermission(this,

                    Manifest.permission.READ_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                        MY_PERMISSIONS_REQUEST_READ_EXTERNAL_STORAGE);

            }
            if (ContextCompat.checkSelfPermission(this,
                    Manifest.permission.CALL_PHONE)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.CALL_PHONE},
                        MY_PERMISSIONS_REQUEST_CALL_PHONE);

            }
        }
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        mContext = this;
        DisplayMetrics displayMetrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        screenHeight = displayMetrics.heightPixels;
        screenWidth = displayMetrics.widthPixels;
        if (screenHeight > screenWidth) {
            int tmp = screenHeight;
            screenHeight = screenWidth;
            screenWidth = tmp;
        }
        setOverlay();
        javaCameraView = (JavaCameraView) findViewById(R.id.java_camera_view);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);
        javaCameraView.setFocusableInTouchMode(true);
        javaCameraView.setFocusable(true);
        javaCameraView.setMaxFrameSize(screenWidth, screenHeight);
        trainningFile = getTrainningFile();
        btnHuongDan = (Button) findViewById(R.id.btnHuongDan);
        btnHuongDan.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                moHuongDan();
            }
        });


    }

    @Override
    protected void onPause() {
        super.onPause();
        detectTimer.cancel();
        if (javaCameraView != null)
            javaCameraView.disableView();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (javaCameraView != null)
            javaCameraView.disableView();

    }

    private int size = 1;

    private double thresholdValue = 35;
    public synchronized void detectCard(Mat input) {
        Mat grayScale = new Mat();
        Imgproc.cvtColor(input, grayScale, Imgproc.COLOR_RGBA2GRAY);
        grayScale = new Mat(grayScale, boudingRect);
        int h = boudingRect.height/4;
        int sampleX = boudingRect.width/2-h;
        int sampleY = boudingRect.height/2-h;
        Rect sampleRect = new Rect(new Point(sampleX, sampleY), new Size(h,h));
        Mat sample = grayScale.submat(sampleRect);
        Core.MinMaxLocResult result = Core.minMaxLoc(sample);
        Imgproc.GaussianBlur(grayScale, grayScale, new Size(3, 3), 0);
        Core.inRange(grayScale, new Scalar(result.minVal-thresholdValue), new Scalar(result.minVal+thresholdValue), grayScale);
        opening(grayScale);
        closing(grayScale);
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(grayScale, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        List<Rect> lstRect = new ArrayList<>();
        for (int i = 0; i < contours.size(); i++) {
            Rect rect = Imgproc.boundingRect(contours.get(i));
            float minHeight = ((float) boudingRect.height) / ((float) 4);
            float maxHeight = ((float) boudingRect.height) / ((float) 10) * ((float) 9);
            float ratio = (float) rect.width / (float) rect.height;
            if (rect.height >= minHeight && rect.height <= maxHeight
                    && ratio >= 0.2 && ratio <= 0.8) {
                lstRect.add(rect);
            }

        }
        if (lstRect.size() > 1) {
            for (int i = 0; i < lstRect.size() - 1; i++) {
                for (int j = i + 1; j < lstRect.size(); j++) {
                    if (lstRect.get(j).x < lstRect.get(i).x) {
                        Rect tmp = lstRect.get(i);
                        lstRect.set(i, lstRect.get(j));
                        lstRect.set(j, tmp);
                    }
                }
            }
        }
        if (lstRect.size() > 1) {
            for (int i = lstRect.size()-1; i >0; i--) {
                Rect current = lstRect.get(i);
                Rect last = lstRect.get(i-1);
                if(last.tl().x<=current.tl().x&&last.br().x>=current.br().x){
                    lstRect.remove(current);
                }
            }
        }

        if (lstRect.size() >= 12) {
            detectTimer.cancel();
            String num = "";
            for (Rect rect : lstRect) {
                try {
                    Mat imgNumber = new Mat(grayScale, rect);
                    opening(imgNumber);
                    closing(imgNumber);
                    Imgproc.threshold(imgNumber, imgNumber, 1, 255, Imgproc.THRESH_BINARY_INV);
                    List<Float> feature = calculate_feature(imgNumber);
                    Mat x = new Mat(1, 32, CvType.CV_32FC1);
                    for (int t = 0; t < feature.size(); t++) {
                        x.put(0, t, feature.get(t));
                    }

                    num += (int) svm.predict(x);

                }catch (Exception e){

                }
            }

            Log.w("Number", "" + num);
            if(num.length()>=12){
                cardNum = num;
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        detectTimer.cancel();
                        showPopup();
                    }
                });

            }
        }

    }
    private void showPopup(){
        //
        //borderView.setVisibility(View.INVISIBLE);
        final Dialog alert = new Dialog(this, R.style.CustomDialog);
        alert.requestWindowFeature(Window.FEATURE_NO_TITLE);
        LayoutInflater inflater = (LayoutInflater) mContext.getSystemService(LAYOUT_INFLATER_SERVICE);
        View vi = inflater.inflate(R.layout.dialog, null);
        final Button btnCall = (Button) vi.findViewById(R.id.btnNap);
        final EditText tvCard = (EditText) vi.findViewById(R.id.tvCardNum);
        tvCard.setText(cardNum);
        final ImageView img = (ImageView) vi.findViewById(R.id.imgView2);
        Bitmap bmp = Bitmap.createBitmap(imgCut.width(), imgCut.height(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(imgCut, bmp);
        img.setImageBitmap(bmp);
        btnCall.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startDetect();
                alert.cancel();
                String encodedHash = Uri.encode("#");
                String call = "*100*"+tvCard.getText().toString()+encodedHash;
                Intent intent = new Intent(Intent.ACTION_CALL);

                intent.setData(Uri.parse("tel:" + call));
                if (ActivityCompat.checkSelfPermission(mContext, Manifest.permission.CALL_PHONE) != PackageManager.PERMISSION_GRANTED) {
                    ActivityCompat.requestPermissions(MainActivity.this,
                            new String[]{Manifest.permission.CALL_PHONE},
                            MY_PERMISSIONS_REQUEST_CALL_PHONE);
                }
                mContext.startActivity(intent);
            }
        });
        Button btnShare = (Button) vi.findViewById(R.id.btnChiaSe);

        btnShare.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                alert.cancel();
                String soCard = cardNum.toString();
                Intent sharingIntent = new Intent(android.content.Intent.ACTION_SEND);
                sharingIntent.setType("text/plain");
                sharingIntent.putExtra(android.content.Intent.EXTRA_SUBJECT, "Card Number :");
                sharingIntent.putExtra(android.content.Intent.EXTRA_TEXT,soCard );
                startActivity(Intent.createChooser(sharingIntent, getResources().getString(R.string.share_using)));
            }
        });

        alert.setContentView(vi);
        DisplayMetrics displaymetrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(displaymetrics);
        alert.getWindow().setLayout(screenWidth/2, screenHeight-screenHeight/6);
        alert.setCanceledOnTouchOutside(false);
        alert.show();
        alert.setOnKeyListener(new DialogInterface.OnKeyListener() {
            @Override
            public boolean onKey(DialogInterface dialog, int keyCode, KeyEvent event) {
                if (keyCode == KeyEvent.KEYCODE_BACK) {
                    dialog.cancel();
                    startDetect();

                    return true;

                }
                return false;
            }
        });

    }
    private void moHuongDan() {
        final Dialog huongDan = new Dialog(this, R.style.CustomDialog);
        huongDan.requestWindowFeature(Window.FEATURE_NO_TITLE);
        LayoutInflater inflater = (LayoutInflater) mContext.getSystemService(LAYOUT_INFLATER_SERVICE);
        View vi = inflater.inflate(R.layout.huong_dan, null);
        huongDan.setContentView(vi);
        DisplayMetrics displaymetrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(displaymetrics);
        huongDan.getWindow().setLayout(screenWidth/2, screenHeight-100);
        huongDan.show();

    }

    List<Float> calculate_feature(Mat img) {

        List<Float> r = new ArrayList<>();
        Imgproc.resize(img, img, new Size(40, 40));
        int h = img.rows() / 4;
        int w = img.cols() / 4;
        int T = img.cols() * img.rows();
        int S = T - Core.countNonZero(img);// s là số pixel màu đen của toàn ảnh
        for (int i = 0; i < img.rows(); i += h) {
            for (int j = 0; j < img.cols(); j += w) {
                Mat cell = new Mat(img, new Rect(i, j, h, w));// lấy từng ô trong ảnh 4x4
                int t = cell.rows() * cell.cols();/// số pixel  trong ô nhỏ.
                int s = t - Core.countNonZero(cell);// sô pixel màu đen.
                float f = ((float) s) / ((float) S);// f là tỷ số pixel màu đen / tổng thể
                r.add(f);
            }
        }
        for (int i = 0; i < 16; i += 4) {// lặp 4 lần.
            float f = r.get(i) + r.get(i + 1) + r.get(i + 2) + r.get(i + 3);// 0 1 2 3 --- > f(0) = 3 ô đầu tiên
            r.add(f);                                                       // i = 4 : 4 5 6 7 --> f4 =
        }

        for (int i = 0; i < 4; ++i) {
            float f = r.get(i) + r.get(i + 4) + r.get(i + 8) + r.get(i + 12);// f(0) = 0 4  8 12
            r.add(f);
        }

        r.add(r.get(0) + r.get(5) + r.get(10) + r.get(15));//1 phần tử
        r.add(r.get(3) + r.get(6) + r.get(9) + r.get(12));// 1 phần tử
        r.add(r.get(0) + r.get(1) + r.get(4) + r.get(5));
        r.add(r.get(2) + r.get(3) + r.get(6) + r.get(7));
        r.add(r.get(8) + r.get(9) + r.get(12) + r.get(13));
        r.add(r.get(10) + r.get(11) + r.get(14) + r.get(15));
        r.add(r.get(5) + r.get(6) + r.get(9) + r.get(10));
        r.add(r.get(0) + r.get(1) + r.get(2) + r.get(3) + r.get(4) + r.get(7) + r.get(8) + r.get(11) + r.get(12) + r.get(13) + r.get(14) + r.get(15));
        return r;
    }

    private void opening(Mat img) {
        Imgproc.erode(img, img, Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, new Size(size, size)));
        Imgproc.dilate(img, img, Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, new Size(size, size)));
    }

    private void closing(Mat img) {
        Imgproc.dilate(img, img, Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, new Size(size, size)));
        Imgproc.erode(img, img, Imgproc.getStructuringElement(Imgproc.MORPH_CROSS, new Size(size, size)));
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()) {
            Log.i(TAG, "Load Thành Công");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
            trainningFile = getTrainningFile();
            svm = SVM.load(trainningFile);
            startDetect();
        } else {
            Log.i(TAG, "Load không thành công");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallback);
        }
        //  borderView.setVisibility(View.VISIBLE);

    }

    private void startDetect() {
        detectTimer = new Timer();
        detectTimer.schedule(new TimerTask() {
            @Override
            public void run() {
                if (mRgba != null) {
                            try {
                                detectCard(mRgba);
                            } catch (Exception e) {

                    }
                }
            }
        }, 0, 700);
    }
    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(width, height, CvType.CV_16S);

    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();

    }

    private String getTrainningFile() {
        try {

            InputStream is = getResources().openRawResource(R.raw.trainning);
            String parentPath = Environment.getExternalStorageDirectory().toString()+"/MobileCardCam";
            File dir = new File(parentPath);
            if(dir.isFile()){
                dir.delete();
            }
            if(!dir.exists()){
                dir.mkdir();
            }
            parentPath += "/trainning.xml";
            File trainningFile = new File(parentPath);
            if(!trainningFile.exists()) {
                FileOutputStream os = new FileOutputStream(parentPath);
                BufferedReader reader = new BufferedReader(new InputStreamReader(is));
                String outString = "";
                String line = "";
                while ((line = reader.readLine()) != null) {
                    outString += line + "\n";
                }
                os.write(outString.getBytes());
                is.close();
                os.flush();
                os.close();
                Log.w("Path", parentPath);
            }
            return parentPath;

        } catch (Exception e) {
            Log.e("OpenCVActivity", "Error loading cascade", e);
        }
        return "";
    }

    static {
        if (OpenCVLoader.initDebug()) {
            Log.d("TAG", "Opencv not loaded");
        } else {
            Log.d("TAG", "Opencv Loaded");
        }
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        x = mRgba.cols() / 2;
        y = mRgba.rows() / 2;
        m = x  / 7;
        Imgproc.line(mRgba, new Point(x - 2.5*m, y - 0.5*m), new Point(x - 2.5*m + 20, y - 0.5*m), new Scalar(255, 255, 255), 6);
        Imgproc.line(mRgba, new Point(x - 2.5*m, y - 0.5*m), new Point(x - 2.5*m, y - 0.5*m + 20), new Scalar(255, 255, 255), 6);
        Imgproc.line(mRgba, new Point(x - 2.5*m, y + 0.5*m), new Point(x - 2.5*m + 20, y + 0.5*m), new Scalar(255, 255, 255), 6);
        Imgproc.line(mRgba, new Point(x - 2.5*m, y + 0.5*m), new Point(x - 2.5*m, y + 0.5*m - 20), new Scalar(255, 255, 255), 6);
        Imgproc.line(mRgba, new Point(x + 2.5*m, y - 0.5*m), new Point(x + 2.5*m - 20, y - 0.5*m), new Scalar(255, 255, 255), 6);
        Imgproc.line(mRgba, new Point(x + 2.5*m, y - 0.5*m), new Point(x + 2.5*m, y - 0.5*m + 20), new Scalar(255, 255, 255), 6);
        Imgproc.line(mRgba, new Point(x + 2.5*m, y + 0.5*m), new Point(x + 2.5*m - 20, y + 0.5*m), new Scalar(255, 255, 255), 6);
        Imgproc.line(mRgba, new Point(x + 2.5*m, y + 0.5*m), new Point(x + 2.5*m, y + 0.5*m - 20), new Scalar(255, 255, 255), 6);

        boudingRect = new Rect((int) (x - 2.5 * m), (int) (y - 0.5 * m), (int) (5 * m), (int) (m));
        imgCut = new Mat(mRgba, boudingRect);
        return mRgba;
    }
        private void setOverlay(){
            width = screenWidth / 2;
            height = screenHeight / 2;
            per = width  / 7;
            boudingRect2 = new Rect((int) (width - 2.5 * per), (int) (height - 0.5 * per), (int) (5 * per), (int) (per));
        View overlayTop = findViewById(R.id.overlayTop);
        View overlayBottom = findViewById(R.id.overlayBottom);
        View overlayLeft = findViewById(R.id.overlayLeft);
        View overlayRight = findViewById(R.id.overlayRight);
        RelativeLayout.LayoutParams params1 = new RelativeLayout.LayoutParams(screenWidth, boudingRect2.y);
        params1.setMargins(0, 0, 0, screenHeight-boudingRect2.y);
        overlayTop.setLayoutParams(params1);
        RelativeLayout.LayoutParams params2 = new RelativeLayout.LayoutParams(screenWidth, boudingRect2.y);
        params2.setMargins(0, boudingRect2.y+boudingRect2.height, 0, 0);
        overlayBottom.setLayoutParams(params2);

        RelativeLayout.LayoutParams params3 = new RelativeLayout.LayoutParams(boudingRect2.x, boudingRect2.height);
        params3.leftMargin = 0;
        params3.topMargin = boudingRect2.y;
        overlayLeft.setLayoutParams(params3);

        RelativeLayout.LayoutParams params4 = new RelativeLayout.LayoutParams(boudingRect2.x, boudingRect2.height);
        params4.leftMargin = boudingRect2.x+boudingRect2.width;
        params4.topMargin = boudingRect2.y;
        overlayRight.setLayoutParams(params4);
    }





}
