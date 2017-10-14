import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;

/*
 * Multilayer perceptron for binary classification.
 */
public class MLP {
	
	ArrayList<double[][]> weights;
	ArrayList<double[]> neuron_vals;
	
	double[] bias;
	int[] layer_sizes;
	
	int hidden_layers;	
	int epochs = 5000;
	double learning_rate;

	 /* Constructor:
	 * int hiddenLayers: number of hidden layers.
	 * int[] layerSizes: number of neurons in each hidden layer.
	 */
	public MLP(int hiddenLayers, int[] layerSizes, double learningRate){
		if(hiddenLayers <= 0 || layerSizes == null || layerSizes.length == 0){System.out.println("Invalid input."); return;}
		else if(layerSizes.length != hiddenLayers){System.out.println("Number of layers must agree with length of layer size vector."); return;}
		else{ // Construct all layers.
			hidden_layers = hiddenLayers;
			layer_sizes = layerSizes;
			weights = new ArrayList<double[][]>(hiddenLayers+1);
			neuron_vals = new ArrayList<double[]>(hiddenLayers+2);
			for(int i = 0 ; i < layerSizes.length ; i++){if(layerSizes[i] <=0){System.out.println("Layer sizes must all be > 0"); return;}}
		}
		if(learningRate <= 0){System.out.println("Learning rate must be positive"); return;}
		learning_rate = learningRate;
		
		System.out.println("Neural network created successfully.");
		
	}
	
	
	// Train the MLP with the provided data.
	public void train(double[][] data, int[] ans){
		if(data == null || data.length == 0 || data[0].length == 0 || ans == null || ans.length != data.length){
			System.out.println("Invalid data dimensions."); return;
		}
		
		// Create weight matrix for input layer
		weights.add(new double[data[0].length][layer_sizes[0]]);
		for(int j = 0 ; j < data[0].length ; j++){
			for(int k = 0 ; k < layer_sizes[0] ; k++){
				weights.get(0)[j][k] =  -1 + (1 - (-1)) * Math.random();
			}
		}
		
		// weight matrices for hidden layer(s)
		for(int i = 0 ; i < hidden_layers-1 ; i++){
			double[][] w = new double[layer_sizes[i]][layer_sizes[i+1]]; 
			for(int j = 0 ; j < layer_sizes[i] ; j++){
				for(int k = 0 ; k < layer_sizes[i+1] ; k++){
					w[j][k] = -1 + (1 - (-1)) * Math.random();
				}
			}
			weights.add(w);
		}
		
		// weight matrix output layer
		double[][] w = new double[layer_sizes[layer_sizes.length-1]][1];
		for(int j = 0 ; j < w.length ; j++){w[j][0] =  -1 + (1 - (-1)) * Math.random();}
		weights.add(w);
		
		// layer values
		neuron_vals.add(new double[data[0].length]); // inputs neuron(s)
		for(int i = 0 ; i < layer_sizes.length ; i++){
			neuron_vals.add(new double[layer_sizes[i]]); // hidden layer neuron(s)
		}
		neuron_vals.add(new double[1]); // output neuron
		
		System.out.println("Training started...");

		for(int ep = 0; ep < epochs ; ep++){
			for(int d = 0 ; d < data.length ; d++){
				for(int i = 0 ; i < data[d].length ; i++){neuron_vals.get(0)[i] = data[d][i];} // initializing input neuron values.
				update(ans[d]); // update all weight matrices.
			}
			if(ep % 1000 == 0 && ep != 0){System.out.println("Completed pass "+ ep);}
		}
		System.out.println("Training complete.");
	}
	
	// Update weights for data point.
	public void update(int ans){
		// Forward pass
		for(int i = 0 ; i <= hidden_layers ; i++){
			double[][] w = weights.get(i);
			for(int j = 0 ; j < w[0].length ; j++){
				double val = 0;
				for(int k = 0 ; k < w.length ; k++){
					val += neuron_vals.get(i)[k] * w[k][j];
				}
				neuron_vals.get(i+1)[j] = activation_func(val); // Update next layer's input values.
			}
		}
		
		// Error calculation if termination depends on it
		//double err = calc_error(neuron_vals.get(neuron_vals.size()-1)[0] , ans);
		
		// Backpropagation
		double[][] prev_errors =  new double[1][1];
		double[][] weight_updates = new double[weights.get(hidden_layers).length][weights.get(hidden_layers)[0].length];

		double output = neuron_vals.get(neuron_vals.size()-1)[0];
		prev_errors[0][0] = ((double)ans - output) * (output*(1-output));
		for(int j = 0 ; j < weight_updates.length ; j++){
			weight_updates[j][0] = prev_errors[0][0] * neuron_vals.get(neuron_vals.size()-2)[j];
		}

		for(int i = hidden_layers-1 ; i >= 0 ; i--){
			double[][] update_weights = new double[weights.get(i).length][weights.get(i)[0].length];
			double[][] errors = new double[weights.get(i+1).length][1];
			
			// Finding weight updates for current layer
			for(int j = 0 ; j < weights.get(i+1).length ; j++){
				double sum = 0;
				for(int k = 0 ; k < weights.get(i+1)[0].length; k++){
					sum += weights.get(i+1)[j][k]*prev_errors[k][0];
				}
				errors[j][0] = sum;
				for(int k = 0 ; k < weights.get(i).length ; k++){
					update_weights[k][j] = sum * (neuron_vals.get(i+1)[j]*(1-neuron_vals.get(i+1)[j])) * (neuron_vals.get(i)[k]);
				}
			}
			
			// Updating weights for previous layer
			for(int j = 0 ; j < weights.get(i+1).length ; j++){
				for(int k = 0 ; k < weights.get(i+1)[0].length ; k++){
					weights.get(i+1)[j][k] += weight_updates[j][k];
				}
			}
			weight_updates = update_weights.clone();
			
			prev_errors = errors.clone(); // New errors to be used for next level.
			
		}
		
		for(int j = 0 ; j < weights.get(0).length ; j++){
			for(int k = 0 ; k < weights.get(0)[0].length ; k++){
				weights.get(0)[j][k] += weight_updates[j][k];
			}
		}
		
		
	}
	
	// Activation function
	private double activation_func(double val){
		return (1.0 / (1+Math.exp(val*-1)));
	}
	// Error calculation
	private double calc_error(double guess, int ans){
		double diff_sq = (double)ans - guess;
		diff_sq *= diff_sq;
		return (0.5 * diff_sq);
	}
	
	// Use the MLP to predict the provided data.
	public int[] predict(double[][] test){
		int[] ans = new int[test.length];
		
		for(int d = 0 ; d < test.length ; d++){
			for(int i = 0 ; i < test[d].length ; i++){neuron_vals.get(0)[i] = test[d][i];} // initializing input neuron values.
			for(int i = 0 ; i <= hidden_layers ; i++){
				double[][] w = weights.get(i);
				for(int j = 0 ; j < w[0].length ; j++){
					double val = 0;
					for(int k = 0 ; k < w.length ; k++){
						val += neuron_vals.get(i)[k] * w[k][j];
					}
					neuron_vals.get(i+1)[j] = activation_func(val); // Update next layer's input values.
				}
			}
			System.out.println(neuron_vals.get(neuron_vals.size()-1)[0]);
			ans[d] = neuron_vals.get(neuron_vals.size()-1)[0] >= 0.5 ? 1 : 0;
		}
		
		return ans;
	}
	
	// Test the accuracy of predicted values.
	public void test_accuracy(int[] sol , int[] predict){
		int num_right = 0;
		for(int i = 0 ; i < sol.length ; i++){
			if(sol[i] == predict[i]){num_right++;}
		}
		
		System.out.println("Accuracy: " + new DecimalFormat("#0.0000").format((double)num_right / (double)sol.length));
	}
	
	// Print hits and misses of a dataset
	private class Pair{
		double x; double y;
		public Pair(double x, double y){
			this.x = x; this.y = y;
		}
	}
	public void print_hits_misses(int[] sol , int[] predict, double[][] data){
		ArrayList<Pair> hits1 = new ArrayList<Pair>();
		ArrayList<Pair> hits0 = new ArrayList<Pair>();
		ArrayList<Pair> misses = new ArrayList<Pair>();
		
		for(int i = 0 ; i < sol.length ; i++){
			if(sol[i] == predict[i] && predict[i] == 1){
				hits1.add(new Pair(data[i][0], data[i][1]));
			}
			else if(sol[i] == predict[i] && predict[i] == 0){
				hits0.add(new Pair(data[i][0], data[i][1]));
			}
			else{
				misses.add(new Pair(data[i][0], data[i][1]));
			}
		}
		 // 2 vectors of hits for x and y.
		System.out.print("hitsX = [");
		for(Pair coord : hits1){System.out.print(coord.x + ", ");}
		System.out.println("];");
		
		System.out.print("hitsY = [");
		for(Pair coord : hits1){System.out.print(coord.y + ", ");}
		System.out.println("];");
		
		System.out.print("hitsX0 = [");
		for(Pair coord : hits0){System.out.print(coord.x + ", ");}
		System.out.println("];");
		
		System.out.print("hitsY0 = [");
		for(Pair coord : hits0){System.out.print(coord.y + ", ");}
		System.out.println("];");
		
		// 2 vectors for misses x and y
		System.out.print("missX = [");
		for(Pair coord : misses){System.out.print(coord.x + ", ");}
		System.out.println("];");
		
		System.out.print("missY = [");
		for(Pair coord : misses){System.out.print(coord.y + ", ");}
		System.out.println("];");
	}
	

}
