vectorImage = imageTo20x20Gray('own_number.jpg',0,0,0); 
p = predict(Theta1, Theta2, vectorImage);
title(strcat("Network hypothesis: ",num2str (mod(p, 10))), "fontsize", 20)
fprintf("Prediction: %d (Number %d)!\n", p,  mod(p, 10));