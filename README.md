# Predicci贸n de Incendios en Galicia
![img](data/Spain_Galicia.png)

## Introducci贸n al Proyecto 馃殌
_Los incendios forestales, de matorrales o vegetaci贸n pueden describirse como cualquier combusti贸n o quema no controlada y no prescrita de plantas en un entorno natural como un bosque, una pradera, etc. Son uno de los mayores problemas ambientales y producen un da帽o ecol贸gico, econ贸mico y humano, irreparables. Por tal motivo una detecci贸n prematura de los mismos es una herramienta vital que puede permitir una lucha m谩s eficaz contra este flagelo._

_Hoy en d铆a los incendios forestales suponen un grave problema ecol贸gico, social y econ贸mico y se trata de un problema que amenaza con intensificarse debido a los efectos del cambio clim谩tico. Poder saber cu谩ndo y d贸nde se producen, as铆 como cu谩l es su extensi贸n, a qu茅 vegetaci贸n afectan y, sobre todo, por qu茅 se producen y qui茅n o qu茅 los causa, es algo fundamental para evitarlos, protegernos de ellos y as铆 poder conservar nuestro patrimonio forestal._

_Cabe destacar que la recuperaci贸n del estrato arb贸reo podr铆a demorar aproximadamente 60 a帽os. Desde el punto de vista de las p茅rdidas econ贸micas derivadas del incendio forestal, se cuentan las p茅rdidas econ贸micas del bosque, el costo de la extinci贸n del incendio y costo parcial de las p茅rdidas de infraestructura. As铆, las personas residentes permanentes del 谩rea quemada, ven perjudicada su calidad de vida, tanto en su salud como en su econom铆a y sus actividades cotidianas, cada vez que acontece dicho siniestro. Por esta raz贸n, un sistema de prevenci贸n del fuego que informe de alertas tempranas podr铆a ayudar a solventar muchas de estas p茅rdidas, alertando a la poblaci贸n y a las autoridades, para que prepare y dirija sus recursos en la posible zona de siniestro, y as铆 evitar al m铆nimo p茅rdidas econ贸micas, ecol贸gicas y principalmente p茅rdidas de vidas humanas._


### Objetivos y tareas馃搵

_Este proyecto tiene como principal objetivo el obtener un mejor entendimiento de t茅cnicas, desarrollo y construcci贸n de un **modelo de machine learning supervisado**. La realizaci贸n de tareas de Data Acquisition, Data Wrangling y EDA nos guiar谩n a la obtenci贸n de un modelo fiable, el cual es nuestra meta final._

_Con nuestro modelo de machine learning pretendemos predecir c贸mo se relacionan las causas de incendios con las distintas variables para poder generar planes eficientes de control del fuego: brindar un sistema de soporte de decisi贸n a la planeaci贸n estrat茅gica de recursos destinados a incendios forestales._

_Crear un modelo que ayude en la predicci贸n de desarrollo de incendios a partir de sus causas har铆a mucho m谩s eficiente la distribuci贸n de los recursos necesarios para la extinci贸n y ayudar铆a tambi茅n en la reducci贸n de costes, da帽os y p茅rdidas. A煤n teniendo en cuenta la gran dificultad que presenta el desarrollo de un modelo de predicci贸n de incendios (por su complejidad y ser altamente no lineal debido a la incertidumbre asociada al comportamiento humano en relaci贸n al fuego), se intentar谩 desarrollar un modelo bas谩ndonos en el hist贸rico de datos de la Regi贸n de Galicia que es la de mayor afluencia de incendios en Espa帽a y de la cual tenemos acceso a un dataset con abundantes registros._
_


### Adquisici贸n de datos (Data Acquisition) :mag:

_Como y de donde obtuvimos el dataset_

_Los datos originales fueron obtenidos de [CIVIO](https://datos.civio.es/dataset/todos-los-incendios-forestales/) y [Aemet](https://opendata.aemet.es/centrodedescargas/productosAEMET), pero estamos utilizando el dataset con un preprocesado mucho m谩s espec铆fico y de un nivel mucho m谩s alto, que crearon [LenaMorianu](https://github.com/LenaMorianu) y su equipo._

### An谩lisis exploratorio de datos (EDA) 馃敩
Llevamos a cabo una investigaci贸n inicial de los datos en b煤squeda de patrones y anomal铆as mediante el c谩lculo de medidas estad铆sticas b谩sicas y gr谩ficas simples.

* _Observamos las primeras y 煤ltimas 5 filas del dataset para tener un vistazo de los datos_
* _Ejecutamos c贸digos para obtener tama帽o y para tipo de dato de cada columna._
* _Buscamos datos nulos._
* _La funci贸n 鈥渄escribe鈥? nos muestra descriptores estad铆sticos b谩sicos de nuestras columnas 鈥渄f.describe().round()鈥漘

Hacer un an谩lisis exploratorio de datos nos permite tener una idea de la distribuci贸n de variables en su conjunto (forma, sesgo, etc) . Utilizamos histogramas, gr谩ficas de l铆nea y boxplot en b煤squeda de:

* _Relaciones 	entre variables._
* _Valores 	at铆picos o puntos inusuales que puedan indicar problemas de calidad de los datos o conducir a descubrimientos interesantes._
* _Patrones 	temporales_

### Conocimiento y Preparaci贸n de datos (Data Wrangling) 馃敡
Ejecutamos c贸digos para ordenamiento y limpieza de nuestra raw data con el fin de detectar y eliminar errores de registro.

* _Eliminamos columnas innecesarias._
* _Verificamos que todos los datos cuenten con el mismo formato: 	Transformamos la columna 'fecha' a tipo numerico para luego dividir mejor las variables._

Una vez establecida la variable causa como variable target de nuestro proyecto, procedimos a la identificaci贸n de las variables m谩s relevantes relacionadas con la primera.

* _Agrupamos dentro del dataset seg煤n la variable causa 鈥?> df.groupby('causa').size()_
* _Mapeamos las variables categoricas que tienen un orden para que sean facilmente adaptables a los modelos._
* _Dado que la variable idmunicipio contiene gran cantidad de 	posibilidades, que decidimos dropear la columna para no agregar varianza a los datos. Latitud y longitud brindan ya la informaci贸n de ubicaci贸n._
* _Dividimos el dataset en dos (datos categoricos y numericos) para optimizar su manejo._
* _Utilizamos catboost para entender la importancia de las variables a la hora de predecir 	la causa. Observamos que a la hora de elegir una variable temporal, nos resulta 	conveniente inclinarnos por 鈥楾rimestre鈥? ya que es mejor predictor por sobre 鈥榤es鈥?._

### Desarrollo del modelo 鈿欙笍

Ya planteadas todas las variables y las relaciones entre ellas, procedimos a crear y elegir el mejor modelo que se ajustara a nuestros objetivos.

* _En primer lugar notamos que la t茅cnica Random Under Sampling fue la que arroj贸 los peores resultados en m茅tricas. Con lo cual podemos afirmar que eliminar datos para intentar balancear las clases no es recomendable para nuestro problema._
* _Es importante complementar las distintas m茅tricas y visualizar la matriz de confusi贸n ya que se puede obtener buenas m茅tricas a pesar de que el modelo no sea 煤til (espec铆ficamente nos referimos al caso de la Maquina de Soporte Vectorial que mencionamos en p谩rrafos anteriores)._
* _Random Forest es el modelo que mejor perfoma en la m茅trica F1 Score; de todas formas notamos que en el caso de aplicaci贸n de las t茅cnicas ROS (mejor F1 observado) y SMOTE, el tiempo requerido es muy alto frente a otros modelos._
* _Por lo tanto, podr铆amos elegir entre dos modelos ganadores: KNN y Random Forest. El primero tiene tiempos muy bajos; A pesar de ello, notamos que (si bien Random Forest no tiene el valor m铆nimo de tiempo, tampoco es demasiado costoso) Random Forest le saca una leve ventaja en F1 (adem谩s de una mayor simpleza a la hora de construirlo) y termina siendo el modelo elegido._

## Herramientas utilizadas en el proyecto 馃洜锔?

_Menciona las herramientas que utilizaste para crear tu proyecto_

* [Anaconda](https://www.anaconda.com/) - La distribuci贸n usada
* [Scikit-learn](https://scikit-learn.org/) - Libreria usada para el modelo de ML
* [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) - Librerias para la visualizaci贸n de gr谩ficos

## Cr茅ditos :handshake:

* Creditos a **Lena Morianu** y a su equipo por el [dataset](https://github.com/LenaMorianu/Los-incendios-en-Galicia) - *Los incendios en Galicia* - [LenaMorianu](https://github.com/LenaMorianu)

## Autores 鉁掞笍

* **Alejandro Nu帽ez** - [AleNunez5](https://github.com/AleNunez5)
* **Carolina Vinagre** - [carovinagre](https://github.com/carovinagre)
* **Claudia Courau** - [clau-courau](https://github.com/clau-courau)
* **Francisco Gutierrez** - [franciscodevs](https://github.com/franciscodevs)

## Tutor :raising_hand_man:
* **Jose Ignacio Lopez Saez** - [nachols1986](https://github.com/nachols1986)


![img](data/LOGO_CODER.png)
