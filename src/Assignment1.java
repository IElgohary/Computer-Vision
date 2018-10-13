import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Assignment1 {
	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		// Q1 ===========================================================================
		Mat source = Imgcodecs.imread("A1I/Q1I1.png", Imgcodecs.CV_LOAD_IMAGE_COLOR);
		Mat batman = Imgcodecs.imread("A1I/Q1I2.jpg", Imgcodecs.CV_LOAD_IMAGE_COLOR);
		
		Mat contrastInc = contrastFading(source);
		Imgcodecs.imwrite("A1I/Answers/Q1/contrast.png", contrastInc);
		
		batman = reflect(batman);
		Mat scaled = scale(source.rows(), source.cols(), batman);
		Mat translated = translate(scaled, 100);
		Imgcodecs.imwrite("A1I/Answers/Q1/batmanTransformed.png", translated);
		
		Mat overlay = overlay(contrastInc, translated);
		Imgcodecs.imwrite("A1I/Answers/Q1/overlay.png", overlay);
		
		// Q2 ===========================================================================
		Mat sherlock = Imgcodecs.imread("A1I/Q2I1.jpg", Imgcodecs.CV_LOAD_IMAGE_COLOR);
		Mat frame1 = Imgcodecs.imread("A1I/Q2I2.jpg", Imgcodecs.CV_LOAD_IMAGE_COLOR);
		Mat frame2 = Imgcodecs.imread("A1I/Q2I3.jpg", Imgcodecs.CV_LOAD_IMAGE_COLOR);
		
		scaled = scale(frame1.rows(), frame1.cols(), sherlock);
		int [][] dimensions1 = {{1218,377}, {1307, 377},{1218,517}};
		Mat transformed = transformAffine(scaled, dimensions1);
		Imgcodecs.imwrite("A1I/Answers/Q2/sherlockFit1.jpg", transformed);
		Mat fitted = fit(transformed, frame1);
		Imgcodecs.imwrite("A1I/Answers/Q2/sherlokFramed1.jpg", fitted);
		
		scaled = scale(frame2.rows(), frame2.cols(), sherlock);
		int [][] dimensions2 = {{371,95}, {702, 128},{327,524}};
		transformed = transformAffine(scaled, dimensions2);
		Imgcodecs.imwrite("A1I/Answers/Q2/sherlockFit2.jpg", transformed);
		fitted = fit(transformed, frame2);
		Imgcodecs.imwrite("A1I/Answers/Q2/sherlokFramed2.jpg", fitted);
		
		// Q3 ===========================================================================
		Mat frame3 = Imgcodecs.imread("A1I/Q3I1.jpg", Imgcodecs.CV_LOAD_IMAGE_COLOR);
		scaled = scale(frame3.rows(), frame3.cols(), sherlock);
		int [][] dimensions3 = {{162,35}, {467, 69},{157,388},{463,352}};
		transformed = transformPerspective(scaled, dimensions3);
		Imgcodecs.imwrite("A1I/Answers/Q3/sherlockFit.jpg", transformed);
		fitted = fit(transformed, frame3);
		Imgcodecs.imwrite("A1I/Answers/Q3/sherlokFramed.jpg", fitted);
	}
	
	
	/**
	 * Apply perspective transformation to an image
	 * @param image
	 * @param dims
	 * @return
	 */
	public static Mat transformPerspective(Mat image, int[][] dims) {
		Mat destination = new Mat();

		Point p1 = new Point(0, 0);
		Point p2 = new Point(image.cols() - 1, 0);
		Point p3 = new Point(0, image.rows() - 1);
		Point p4 = new Point(image.cols() - 1 , image.rows() - 1);
		Point p5 = new Point(dims[0][0] , dims[0][1]);
		Point p6 = new Point(dims[1][0] , dims[1][1]);
		Point p7 = new Point(dims[2][0] , dims[2][1]);
		Point p8 = new Point(dims[3][0] , dims[3][1]);
		
		MatOfPoint2f ma1 = new MatOfPoint2f(p1, p2, p3, p4);
		MatOfPoint2f ma2 = new MatOfPoint2f(p5, p6, p7, p8);
		Mat tranformMatrix = Imgproc.getPerspectiveTransform(ma1, ma2);
		
		Size size = new Size(image.cols(), image.rows());
		Imgproc.warpPerspective(image, destination, tranformMatrix, size);
		return destination;
	}
	
	/**
	 * Fit an image inside a frame
	 * @param image
	 * @param frame
	 * @return
	 */
	public static Mat fit(Mat image, Mat frame) {
		int rows = image.rows();
		int cols = image.cols();
		int ch = image.channels();
		Mat destination = new Mat(rows, cols, image.type());
		
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				double[] imageData = image.get(i, j);
				double[] frameData = frame.get(i, j);
				if(imageData[0] == 0 && imageData[1] == 0 && imageData[2] == 0) {
					imageData[0] = frameData[0];
					imageData[1] = frameData[1];
					imageData[2] = frameData[2];
				}
				destination.put(i, j, imageData); 
			}
		}
		return destination;
	}
	
	/**
	 * Apply affine transformation to a matrix
	 * @param image
	 * @param dims 2d array of 3 points/ coordinates
	 * @return
	 */
	public static Mat transformAffine(Mat image, int[][] dims) {
		Mat destination = new Mat();

		Point p1 = new Point(0, 0);
		Point p2 = new Point(image.cols() - 1, 0);
		Point p3 = new Point(0, image.rows() - 1);
		Point p4 = new Point(dims[0][0] , dims[0][1]);
		Point p5 = new Point(dims[1][0] , dims[1][1]);
		Point p6 = new Point(dims[2][0] , dims[2][1]);
		
		MatOfPoint2f ma1 = new MatOfPoint2f(p1, p2, p3);
		MatOfPoint2f ma2 = new MatOfPoint2f(p4, p5, p6);
		Mat tranformMatrix = Imgproc.getAffineTransform(ma1, ma2);
		
		Size size = new Size(image.cols(), image.rows());
		Imgproc.warpAffine(image, destination, tranformMatrix, size);
		return destination;
	}
	
	/**
	 * Translate an image on the X-axis by the value of translateBy 
	 * @param image
	 * @param translateBy
	 * @return
	 */
	public static Mat translate(Mat image, int translateBy) {
		Mat destination = new Mat();

		Point p1 = new Point(0, 0);
		Point p2 = new Point(image.cols() - 1, 0);
		Point p3 = new Point(0, image.rows() - 1);
		Point p4 = new Point(0 + translateBy, 0);
		Point p5 = new Point(image.cols() - 1 + translateBy, 0);
		Point p6 = new Point(0 + translateBy, image.rows() - 1);

		MatOfPoint2f ma1 = new MatOfPoint2f(p1, p2, p3);
		MatOfPoint2f ma2 = new MatOfPoint2f(p4, p5, p6);
		Mat tranformMatrix = Imgproc.getAffineTransform(ma1, ma2);

		Size size = new Size(image.cols(), image.rows());
		Imgproc.warpAffine(image, destination, tranformMatrix, size);
		return destination;
	}
	
	/**
	 * Blend 2 images together
	 * @param image1
	 * @param image2
	 * @return
	 */
	public static Mat overlay(Mat image1, Mat image2) {
		int rows = image1.rows();
		int cols = image1.cols();
		int ch = image1.channels();
		Mat destination = new Mat(rows, cols, image1.type());
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				double[] image1Data = image1.get(i, j);
				double[] image2Data = image2.get(i, j);
				for (int k = 0; k < ch; k++) {
					image1Data[k] = (0.7 * image1Data[k] + 0.3 * image2Data[k]);
				}
				destination.put(i, j, image1Data); 
			}
		}
		return destination;
	}

	public static Mat scale(int rows, int cols, Mat image) {
		Mat destination = new Mat();
		Size sz = new Size(cols, rows);
		Imgproc.resize(image, destination, sz);
		return destination;
	}
	
	/**
	 * Reflect an image over the Y-axis
	 * @param image
	 * @return
	 */
	public static Mat reflect(Mat image) {
		int rows = image.rows();
		int cols = image.cols();
		int ch = image.channels();
		Mat destination = new Mat(rows, cols, image.type());

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				double[] data;
				if (j < cols / 2) {
					data = image.get(i, cols - j - 1);
				} else {
					data = image.get(i, cols - j - 1);
				}
				double[] result = image.get(i, j);
				for (int k = 0; k < ch; k++) {
					result[k] = data[k];

				}
				destination.put(i, j, result);
			}
		}
		return destination;
	}
	
	/**
	 * Increase contrast of an image but effect
	 * fades towards the right 
	 * 
	 * @param image
	 * @return
	 */
	public static Mat contrastFading(Mat image) {
		int rows = image.rows();
		int cols = image.cols();
		int ch = image.channels();
		Mat destination = new Mat(rows, cols, image.type());
		float contrastFactor = 3;
		float reduceBy = (float) (contrastFactor / (cols / 2));

		for (int i = 0; i < rows; i++) {
			contrastFactor = 3;
			for (int j = 0; j < cols; j++) {
				double[] data = image.get(i, j);
				for (int k = 0; k < ch; k++) {
					if (j < cols / 2)
						data[k] = data[k] * contrastFactor;
					else {
						contrastFactor -= reduceBy;
						data[k] = data[k] * contrastFactor;
					}
				}
				destination.put(i, j, data);
			}
		}
		return destination;
	}
}
