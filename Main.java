import java.util.Random;
/*
 * Here we create a multilayer perceptron classifier.  First we do a simple linear classification.  Next, we attempt to classify points
 * inside and outside a unit circle.
 */
public class Main {
	
	
	public static void main(String[] args){
		int[] layer_size = {3};
		MLP mlp_classifier = new MLP(1, layer_size, 0.01);
		
		double[][] train_data = new double[100000][4];
		int[] train_ans = new int[100000];
		
		Random r = new Random();
		// Obtain some 2D Cartesian points for training.
		for(int i = 0 ; i < 100000 ; i++){
			// Line
			/*double x = r.nextDouble();
			double y = r.nextDouble();*/
			
			// Circle
			double x = -2 + (2 - (-2)) * r.nextDouble();
			double y = -2 + (2 - (-2)) * r.nextDouble();
			
			train_data[i][0] = x;
			train_data[i][1] = y;
			train_data[i][2] = x*x;
			train_data[i][3] = y*y;
			
			//int a = y >= x ? 1 : 0; // 1 if above line y=x, 0 otherwise.
			int a = x*x+y*y <= 1 ? 1 : 0;  // 1 if in unit circle, 0 o.w.
			
			train_ans[i] = a;
			
		}
		
		mlp_classifier.train(train_data, train_ans);
		
		double[][] test_data = new double[1000][4];
		int[] test_ans = new int[1000];

		for(int i = 0 ; i < 1000 ; i++){
			//Line
			/*double x = r.nextDouble();
			double y = r.nextDouble();*/

			// Circle
			double x = -2 + (2 - (-2)) * r.nextDouble();
			double y = -2 + (2 - (-2)) * r.nextDouble();

			test_data[i][0] = x;
			test_data[i][1] = y;
			test_data[i][2] = x*x;
			test_data[i][3] = y*y;

			//int a = y >= x ? 1 : 0; // 1 if above line y=x, 0 otherwise.
			int a =  x*x+y*y <= 1 ? 1 : 0;  // 1 if in unit circle, 0 o.w.
			
			test_ans[i] = a;
			
		}
		
		int[] pred = mlp_classifier.predict(test_data);
		mlp_classifier.test_accuracy(test_ans, pred);
		mlp_classifier.print_hits_misses(test_ans, pred, test_data);
	}

}
