# Deep Learning Model for Trading
K-line pattern recognition tool capable of predicting future price movements. Requires input of price sequences (recommended length: 20-100).<br>
## Description<br>
Utilizes a two-stream LSTM model (price and price difference) to identify K-line patterns:<br>

W1 (Higher Left, Lower Right): Predicts a downtrend.<br>
W2 (Lower Left, Higher Right): May indicate either an uptrend or downtrend. Included to improve the model's accuracy in distinguishing W1 patterns.<br>
M1 (Lower Left, Higher Right): Predicts an uptrend.<br>
M2 (Higher Left, Lower Right): May indicate either an uptrend or downtrend. Included to improve the model's accuracy in distinguishing M1 patterns.<br>
X1 (V-shaped, inverse V-shaped, sideways movements): Included to enhance overall classification accuracy.<br>
Unknown: Patterns with insufficient confidence for classification.<br>

## Model Classification Names:
class_names = <br>
['w1_down', 'm1_up', 'w2_up_or_down', 'm2_up_or_down', 'trend_unknown', 'uncertain'] <br>

## Trading Strategy
Short positions after detecting a W1 pattern in a downtrend. <br>
Long positions after detecting an M1 pattern in an uptrend. <br>

## Upgrade Plan
Future updates will incorporate trading volume data. Current price-only predictions already yield satisfactory results. <br>
(Manually labeling training samples is time-consuming...) <br>

## Upgrade Plan
If you're interested in naked K-line strategies or have specific requirements, contact: <br>
email: 321954771@qq.com <br>

## Other Description
Little Fucking Japanese get Fucking out Here!  <br>
Dont use the tools to earn your fucking money  <br>
