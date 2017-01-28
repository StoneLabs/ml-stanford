function p = predict_own(path, invert = 0)
  %% Setup the parameters you will use for this exercise
  input_layer_size  = 400;  % 20x20 Input Images of Digits
  hidden_layer_size = 25;   % 25 hidden units
  num_labels = 10;          % 10 labels, from 1 to 10   
                            % (note that we have mapped "0" to label 10)

  load('ex3data1.mat');
  m = size(X, 1);

  % fprintf('\nLoading Saved Neural Network Parameters ...\n')
  load('ex3weights.mat');
  
  % fprintf('\nPredicting ...\n')
  vectorImage = imageTo20x20Gray(path,0,0,invert); 
  p = predict(Theta1, Theta2, vectorImage);
  title(strcat("Network hypothesis: ",num2str (mod(p, 10))), "fontsize", 20)
  % fprintf("Prediction: %d (Number %d)!\n", p,  mod(p, 10));
end