# Модуль 10. Згорткові нейронні мережі. Tensorflow. Keras.

*З циклу [домашніх завдань Python Data Science](https://github.com/lexxai/goit_python_data_sciense_homework).*

# Домашнє завдання

## Частина 1

В якості домашнього завдання вам пропонується створити нейронну мережу за допомогою механізмів Keras, яка буде класифікувати товари із датасету [fasion_mnist](https://www.tensorflow.org/datasets/catalog/fashion_mnist)[fasion_mnist](https://www.tensorflow.org/datasets/catalog/fashion_mnist).

На відміну від попереднього завдання вам пропонується створити згорткову нейромережу. Підберіть архітектуру мережі та навчіть її на даних із датасету fasion_mnist. Спробуйте досягти максимально можливої точності класифікації за рахунок маніпуляції параметрами мережі. Порівняйте точність отриманої згорткової мережі з точністю багатошарової мережі з попереднього завдання. Зробіть висновки.

## Частина 2

В цій частині ми знову будемо працювати з датасетом [fasion_mnist](https://www.tensorflow.org/datasets/catalog/fashion_mnist).

На відміну від попереднього завдання вам пропонується створити згорткову нейромережу, що використовує VGG16 в якості згорткової основи.

Навчіть отриману мережу на даних із датасету fasion_mnist. Спробуйте досягти максимально можливої точності класифікації за рахунок маніпуляції параметрами мережі. Під час навчання використовуйте прийоми донавчання та виділення ознак.

Порівняйте точність отриманої згорткової мережі з точністю багатошарової мережі з попереднього завдання. Зробіть висновки.


# Результати

- [goit_python_ds_hw_10.ipynb](goit_python_ds_hw_10.ipynb)
- [Colab (goit_python_ds_hw_10.ipynb)](https://drive.google.com/file/d/1UZ0mGqIN2Rqcs3OxCcwLOTiLSasm5oT8/view?usp=sharing)
- [models.zip](https://drive.google.com/file/d/1A1bqwSmqWNuChNcpw9SsQm91-AXhxKRZ/view?usp=drive_link)

## Models compare
![va](img_va.png)

## Image Predict
![img_predict](img_predict.png)


## Image Generator
![image_genarator](image_genarator.png)


## Learning curves




![Learning of (Model_1)](img_Model_1_ca.png)

![Learning of (Model_2)](img_Model_2_ca.png)

![Learning of (Model_2_s2)](img_Model_2_s2_ca.png)

![Learning of (Model_2_L2_0.001)](img_Model_2_L2_0.001_ca.png)

![Learning of (Model_2_L2_0.05_lr_0.025e)](img_Model_2_L2_0.05_lr_0.025e_ca.png)

![Learning of (Model_2_L1_0.001_B_256)](img_Model_2_L1_0.001_B_256_ca.png)

![Learning of (Model_3)](img_Model_3_ca.png)

![Learning of (Model_4)](img_Model_4_ca.png)

![Learning of (Model_5_VGG16)](img_Model_5_VGG16_ca.png)

![Learning of (Model_6_VGG16)](img_Model_6_VGG16_ca.png)

![Learning of (Model_7_VGG16)](img_Model_7_VGG16_ca.png)

![Learning of (Model_8_VGG16)](img_Model_8_VGG16_ca.png)

![Learning of (Model_9_VGG16)](img_Model_9_VGG16_ca.png)

![Learning of (Model_9_VGG16_SC)](img_Model_9_VGG16_SC_ca.png)

![Learning of (Model_10_VGG16_AL)](img_Model_10_VGG16_AL_ca.png)

![Learning of (Model_2_s2_igen)](img_Model_2_s2_igen_ca.png)

![Learning of (Model_2_s2_igen_c)](img_Model_2_s2_igen_c_ca.png)

![Learning of (Model_LeNet_5)](img_Model_LeNet_5_ca.png)



## Confusion matrix

| Model  | Model  | Model  |
|:------:|:------:|:------:|
| <img src="img_Model_1_cm.png" alt="Confusion matrix of (Model_1)" width="235"> | <img src="img_Model_2_cm.png" alt="Confusion matrix of (Model_2)" width="235"> | <img src="img_Model_2_s2_cm.png" alt="Confusion matrix of (Model_2_s2)" width="235"> |
| Model_1 | Model_2 | Model_2_s2 |
| <img src="img_Model_2_L2_0.001_cm.png" alt="Confusion matrix of (Model_2_L2_0.001)" width="235"> | <img src="img_Model_2_L2_0.05_lr_0.025e_cm.png" alt="Confusion matrix of (Model_2_L2_0.05_lr_0.025e)" width="235"> | <img src="img_Model_2_L1_0.001_B_256_cm.png" alt="Confusion matrix of (Model_2_L1_0.001_B_256)" width="235"> |
| Model_2_L2_0.001 | Model_2_L2_0.05_lr_0.025e | Model_2_L1_0.001_B_256 |
| <img src="img_Model_3_cm.png" alt="Confusion matrix of (Model_3)" width="235"> | <img src="img_Model_4_cm.png" alt="Confusion matrix of (Model_4)" width="235"> | <img src="img_Model_6_VGG16_cm.png" alt="Confusion matrix of (Model_6_VGG16)" width="235"> |
| Model_3 | Model_4 | Model_6_VGG16 |
| <img src="img_Model_7_VGG16_cm.png" alt="Confusion matrix of (Model_7_VGG16)" width="235"> | <img src="img_Model_8_VGG16_cm.png" alt="Confusion matrix of (Model_8_VGG16)" width="235"> | <img src="img_Model_9_VGG16_cm.png" alt="Confusion matrix of (Model_9_VGG16)" width="235"> |
| Model_7_VGG16 | Model_8_VGG16 | Model_9_VGG16 |
| <img src="img_Model_9_VGG16_SC_cm.png" alt="Confusion matrix of (Model_9_VGG16_SC)" width="235"> | <img src="img_Model_10_VGG16_AL_cm.png" alt="Confusion matrix of (Model_10_VGG16_AL)" width="235"> | <img src="img_Model_2_s2_igen_cm.png" alt="Confusion matrix of (Model_2_s2_igen)" width="235"> |
| Model_9_VGG16_SC | Model_10_VGG16_AL | Model_2_s2_igen |
| <img src="img_Model_2_s2_igen_c_cm.png" alt="Confusion matrix of (Model_2_s2_igen_c)" width="235"> | <img src="img_Model_LeNet_5_cm.png" alt="Confusion matrix of (Model_LeNet_5)" width="235"> |  |
| Model_2_s2_igen_c | Model_LeNet_5 |  |

