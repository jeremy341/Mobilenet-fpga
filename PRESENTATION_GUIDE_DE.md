# Präsentations- und Erklärleitfaden

Dieser Leitfaden erklärt das Repository jeremy341/Mobilenet-fpga: Ziel, Grundlagen, Datenpipeline, MobileNetV1, MobileNetV2, Quellcode, Quantisierung, Netron, Ergebnisse und Einschränkungen.

## 1. Kurzfassung

> In diesem Projekt implementiere und vergleiche ich MobileNetV1 und MobileNetV2 für Bildklassifikation. Beide Modelle verwenden den Input 128×128×3 und den Width Multiplier alpha=0.5. MobileNetV1 reduziert Kosten durch Depthwise-Separable Convolutions. MobileNetV2 erweitert dieses Prinzip um Inverted Residual Blocks und Linear Bottlenecks. Ich trainiere zuerst beide Modelle in Float32, konvertiere sie danach mit einer Representative Dataset in vollständig quantisierte INT8-TFLite-Modelle und vergleiche Accuracy, Parameteranzahl, Dateigröße und Datentypen. FPGA-Implementierung ist nicht Teil der Aufgabe.

Gesamtablauf:

~~~text
Dataset → Split → Resize/Normalisierung
→ MobileNetV1 und MobileNetV2
→ Float32-Training → Float32-Evaluation
→ TFLite-Konvertierung → INT8-Evaluation
→ Netron- und Bitbreitenprüfung → Vergleich
~~~

## 2. Repository

~~~text
PROJECT_AIMS.md       Anforderungen und Lern-Roadmap
README.md             Projektübersicht und Befehle
NOTES.md              aktuelle Experimentnotizen
pyproject.toml        Abhängigkeiten
uv.lock               gelockte Abhängigkeitsversionen
check_dataset.py      Dataset-Smoke-Test
src/data.py           Daten laden und vorbereiten
src/models.py         V1- und V2-Architektur
src/train_v1.py       V1-Training
src/train_v2.py       V2-Training
src/quantize.py       Float32-zu-INT8-Konvertierung
src/evaluate.py       Evaluation
results/*.tflite      quantisierte Modelle
results/comparison.md Vergleichswerte
~~~

## 3. Grundlagen

Ein Tensor ist ein mehrdimensionales Zahlenarray. Ein RGB-Bild hat die Form (128,128,3): Höhe, Breite, Kanäle. Eine Batch von 128 Bildern hat die Form (128,128,128,3).

Weights sind lernbare Parameter. Filter einer Convolution sind Weights. Ein Bias ist ein lernbarer Offset:

~~~text
Output = Input × Weight + Bias
~~~

Der Forward Pass schickt ein Bild durch alle Layer. Die Loss misst den Fehler. Backpropagation berechnet Gradienten. Der Optimizer verändert die Weights.

~~~text
Forward Pass → Loss → Gradienten → Optimizer-Update
~~~

Training-Daten verändern die Weights. Validation-Daten überwachen die Entwicklung. Test-Daten werden erst am Ende für die finale Bewertung verwendet.

Eine Batch enthält mehrere Bilder. Eine Epoch ist ein kompletter Durchlauf durch die Trainingsdaten. Adam mit Learning Rate 0.001 wird im aktuellen Code verwendet.

## 4. Dataset und Pipeline

CIFAR-10 besitzt 50.000 Trainingsbilder, 10.000 Testbilder, zehn Klassen und ursprüngliche Bilder der Form 32×32×3. Der Code teilt die Trainingsdaten reproduzierbar in 45.000 Training und 5.000 Validation.

Die Vorverarbeitung ist:

~~~text
32×32×3 → Resize auf 128×128×3
→ Float32 → Division durch 255
→ Pixelbereich 0..1
→ One-Hot-Labels
→ Batching → Prefetching
~~~

Das Resize erzeugt keine neue Bildinformation, erfüllt aber die geforderte Inputgröße.

## 5. src/data.py

Datei: src/data.py

~~~python
import numpy as np
import tensorflow as tf
~~~

NumPy wird für den reproduzierbaren Split verwendet. TensorFlow liefert Dataset, Resizing, One-Hot-Encoding und Training.

~~~python
IMG_SIZE = 128
SEED = 67
VAL_SIZE = 5000
NUM_CLASSES = 10
~~~

Definiert Inputgröße, Seed, Validation-Größe und Klassenanzahl.

~~~python
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
~~~

Lädt CIFAR-10.

~~~python
y_train, y_test = y_train.squeeze(1), y_test.squeeze(1)
~~~

Entfernt die unnötige Label-Dimension von Shape (N,1) zu (N,).

~~~python
rng = np.random.default_rng(SEED)
indices = rng.permutation(x_train.shape[0])
~~~

Erzeugt eine reproduzierbare zufällige Indexreihenfolge.

~~~python
val_idx, train_idx = indices[:VAL_SIZE], indices[VAL_SIZE:]
~~~

Erste 5.000 Indizes gehen in Validation, die restlichen 45.000 ins Training.

~~~python
return ((x_train[train_idx], y_train[train_idx]),
        (x_train[val_idx], y_train[val_idx]),
        (x_test, y_test))
~~~

Gibt Training, Validation und Test zurück.

~~~python
def make_dataset(x, y, batch_size=128, train=False):
~~~

Erzeugt eine TensorFlow-Datenpipeline mit Batchgröße 128.

~~~python
def prepare(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, "float32") / 255.0
~~~

Resized das Bild und normiert die Pixel.

~~~python
label = tf.one_hot(tf.cast(label, tf.int32), NUM_CLASSES)
label = tf.cast(label, "float32")
return image, label
~~~

Erzeugt ein One-Hot-Label mit zehn Werten und gibt Bild plus Label zurück.

~~~python
ds = tf.data.Dataset.from_tensor_slices((x, y))
if train:
    ds = ds.shuffle(10000, seed=SEED)
ds = ds.batch(batch_size)
ds = ds.map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
return ds.prefetch(tf.data.AUTOTUNE)
~~~

Erstellt das Dataset, shuffelt nur Training, bildet Batches, verarbeitet Beispiele parallel und prefetcht kommende Batches.

## 6. CNN-Layer

Conv2D wendet gelernte Filter an. Ein Filter erkennt lokale Muster wie Kanten oder Texturen. Frühe Layer lernen einfache Muster; tiefere Layer kombinieren sie zu Formen.

Conv2D(32, 3, strides=2, padding="same") bedeutet 32 Filter, Kernel 3×3, Stride 2 und Rand-Padding. Dadurch wird 128×128 ungefähr zu 64×64 und die Ausgabe besitzt 32 Kanäle.

BatchNormalization stabilisiert Aktivierungen und besitzt lernbare Skalierung und Verschiebung. Deshalb wird bei Convolutions häufig use_bias=False gesetzt.

ReLU ist max(0,x). Sie entfernt negative Werte und fügt Nichtlinearität hinzu.

ReLU6 ist min(max(0,x),6). Sie begrenzt Aktivierungen auf 0 bis 6 und wird in V2 genutzt.

GlobalAveragePooling2D mittelt jede Feature Map räumlich, zum Beispiel 4×4×512 zu 512. Das spart Parameter gegenüber Flatten plus großem Dense-Layer.

Dense(10) erzeugt zehn Klassenscores. Softmax wandelt sie in klassenähnliche Wahrscheinlichkeiten um.

## 7. MobileNetV1

Eine normale Convolution verbindet gleichzeitig räumliche Positionen und Kanäle. Ihre Kosten sind ungefähr:

~~~text
K² × M × N × D²
~~~

MobileNetV1 ersetzt sie durch:

~~~text
Depthwise 3×3 → Pointwise 1×1
~~~

Depthwise verarbeitet jeden Kanal separat räumlich. Pointwise mischt die Kanäle mit einem 1×1-Filter.

Die Kosten werden ungefähr:

~~~text
K² × M × D² + M × N × D²
~~~

Das ist wesentlich günstiger.

### V1-Block

~~~python
def _depthwise_block(x, filters, stride):
    x = tf.keras.layers.DepthwiseConv2D(
        3, strides=stride, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
~~~

Depthwise-Convolution, Normalisierung und ReLU.

~~~python
    x = tf.keras.layers.Conv2D(
        filters, 1, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x
~~~

Pointwise-Convolution, Normalisierung, ReLU und Rückgabe des Blocks.

### V1-Modell

Der Input wird durch eine normale initiale Convolution von 128×128×3 auf 64×64×32 gebracht. Danach folgen 13 Depthwise-Separable-Blöcke.

Die Liste blocks enthält Tupel:

~~~text
(Outputkanäle vor alpha, Stride, Wiederholungen)
~~~

Die Schleife multipliziert die Outputkanäle mit alpha, erzeugt die Wiederholungen und setzt nach dem ersten Block einer Stage den Stride auf 1. So downsamplet nur der erste Block einer wiederholten Stage.

Der räumliche Verlauf ist:

~~~text
128×128×3
→ 64×64×32
→ 64×64×32
→ 32×32×64
→ 16×16×128
→ 8×8×256
→ 4×4×512
→ GlobalAveragePooling → 512
→ Dense(10)
~~~

## 8. MobileNetV2

V2 verwendet ebenfalls Depthwise-Convolutions, aber zusätzlich:

- Expansion
- Inverted Residuals
- Linear Bottlenecks
- Residual Connections
- ReLU6

Ein Block ist:

~~~text
schmaler Input
→ 1×1 Expansion
→ breite Zwischenrepräsentation
→ Depthwise 3×3
→ 1×1 lineare Projection
→ optionaler Skip
~~~

Bei Expansion Ratio 6 werden beispielsweise 16 Kanäle intern auf 96 erweitert. Die Projection komprimiert wieder.

Ein Residual ist nur möglich, wenn Stride 1 gilt und Input- sowie Outputkanäle identisch sind:

~~~python
if stride == 1 and channels == filters:
    x = tf.keras.layers.Add()([shortcut, x])
~~~

Der Linear Bottleneck besitzt keine ReLU nach der Projection. Das verhindert, dass Informationen in der schmalen Repräsentation verloren gehen.

Die V2-Liste besitzt Tupel:

~~~text
(Expansion Ratio, Outputkanäle vor alpha, Blockanzahl, Stride des ersten Blocks)
~~~

Der räumliche Verlauf ist ungefähr:

~~~text
128×128×3
→ 64×64×32
→ 64×64×8
→ 32×32×12
→ 16×16×16
→ 8×8×32
→ 8×8×48
→ 4×4×80
→ 4×4×160
→ 4×4×640
→ GlobalAveragePooling → 640
→ Dense(10)
~~~

## 9. src/models.py: V2-Code

Die Funktion _inverted_residual_block liest mit x.shape[-1] die Inputkanäle und speichert x als shortcut. Bei expand_ratio ungleich 1 wird eine 1×1-Convolution mit channels × expand_ratio erzeugt, normalisiert und mit ReLU6 aktiviert.

Danach folgt die Depthwise-3×3-Convolution mit dem gewünschten Stride, BatchNormalization und ReLU6. Anschließend projiziert eine 1×1-Convolution auf filters. Nach dieser Projection gibt es nur BatchNormalization, keine ReLU: das ist der Linear Bottleneck.

build_mobilenet_v2 definiert Input, initiale Convolution, die sieben Stages, die finale 1×1-Convolution mit 1280×alpha Kanälen, GlobalAveragePooling und den Zehn-Klassen-Classifier.

Beide Builder erzeugen mit tf.keras.Model einen Keras-Graph und konfigurieren Adam, Categorical Cross-Entropy und Accuracy.

## 10. alpha=0.5

Der Width Multiplier reduziert Kanalzahlen:

~~~python
filters = int(filters * alpha)
~~~

Beispiel: 256 Kanäle werden zu 128. Das reduziert meist Parameter, Speicher und Rechenaufwand, kann aber Accuracy kosten.

Die initiale Conv2D wird jetzt ebenfalls mit alpha multipliziert. Bei alpha=0.5 besitzt sie 16 Filter. Die bisherigen Kaggle-Artefakte wurden jedoch vor dieser Korrektur erzeugt und müssen deshalb neu trainiert und neu quantisiert werden.

## 11. Training

train_v1.py und train_v2.py importieren Path, die Datenfunktionen und den jeweiligen Model Builder.

ROOT findet mit Path(__file__).resolve().parent.parent das Projektverzeichnis. MODELS_DIR zeigt auf models.

load_data lädt die Splits. make_dataset erzeugt Training und Validation. Die Testdaten werden beim Training bewusst nicht verwendet.

build_mobilenet_v1 beziehungsweise build_mobilenet_v2 erzeugt das Modell. model.fit trainiert mit Training-Dataset, prüft nach jeder Epoch auf Validation und verwendet aktuell fünf Epochs.

MODELS_DIR.mkdir legt den Speicherordner an. model.save schreibt die Float32-Datei mobilenet_v1.keras beziehungsweise mobilenet_v2.keras.

Für ein endgültiges Ergebnis wären eher 30–50 maximale Epochs, Early Stopping und Learning-Rate-Reduktion sinnvoll. Beide Modelle müssen dabei gleich behandelt werden.

## 12. src/quantize.py

representative_dataset lädt Trainingsbilder, nimmt 200 Beispiele, resized sie auf 128×128, normiert auf 0–1, fügt die Batch-Dimension hinzu und liefert sie mit yield an den Converter.

Die Representative Dataset ist für die Kalibrierung der Aktivierungsbereiche nötig. Der Converter beobachtet reale Beispiele und bestimmt geeignete Scales und Zero-Points.

TFLiteConverter.from_keras_model lädt das Keras-Modell in den Converter. Optimize.DEFAULT aktiviert Optimierungen. representative_dataset übergibt die Kalibrierung.

TFLITE_BUILTINS_INT8 fordert INT8-Operatoren. inference_input_type und inference_output_type setzen die externe Schnittstelle auf INT8.

convert erzeugt die TFLite-Binärdatei. mkdir erzeugt den Ergebnisordner, write_bytes schreibt die Datei. Der Main-Block wiederholt den Vorgang für V1 und V2.

## 13. Quantisierung

INT8 stellt signed Werte von -128 bis 127 dar. Die Beziehung ist:

~~~text
real = scale × (quantized - zero_point)
quantized = real / scale + zero_point
~~~

Ein quantisierter Convolution-Pfad ist:

~~~text
INT8 Input × INT8 Weight
→ Produkt
→ viele Produkte als INT32 akkumulieren
→ Requantisierung
→ INT8 Output
~~~

INT32 wird benötigt, weil bereits 127×127=16129 ist und viele Produkte addiert werden.

## 14. src/evaluate.py

evaluate_keras lädt ein Keras-Modell und ruft model.evaluate auf. Das gibt Loss und Accuracy zurück.

evaluate_tflite erstellt einen tf.lite.Interpreter und ruft allocate_tensors auf. get_input_details und get_output_details lesen Metadaten. Aus quantization werden Scale und Zero-Point gelesen.

Für jedes Testbild wiederholt der Code Resize und Normalisierung. Danach wird mit img / input_scale + input_zero nach INT8 quantisiert.

set_tensor setzt den Input. invoke führt Inferenz aus. get_tensor liest den Output. Der Output wird mit der inversen Quantisierungsformel nach Float32 zurückgerechnet.

argmax wählt die Klasse mit dem größten Wert. Die Accuracy ist:

~~~text
korrekte Predictions / Anzahl Testbilder
~~~

## 15. Netron-Befunde

Für V1 zeigte Netron:

~~~text
Input:  int8[-1,128,128,3]
Output: int8[-1,10]
~~~

Die Layerfolge war:

~~~text
Conv2D
→ DepthwiseConv2D
→ Conv2D 1×1
→ DepthwiseConv2D
→ ...
→ Mean
→ FullyConnected
→ Softmax
~~~

Beispiele:

~~~text
Filter: int8[32,3,3,3]
Bias:   int32[32]
Final weights: int8[10,512]
Final bias:    int32[10]
~~~

Die Input-Quantisierung wurde ungefähr als 0.003921568859368563 × (q + 128) angezeigt. Das entspricht scale ungefähr 1/255 und zero_point -128.

In Netron prüft man zuerst Graph Properties, dann einzelne Operatoren und Tensoren. Man klickt eine Conv2D, eine DepthwiseConv2D, einen Filter, einen Bias und den Classifier an. Entscheidend ist, die tatsächlichen Tensor-Datentypen zu lesen, nicht nur den Dateinamen.

## 16. Ergebnisse

Die bisher dokumentierten Werte:

| Messgröße | V1 Float32 | V1 INT8 | V2 Float32 | V2 INT8 |
|---|---:|---:|---:|---:|
| Test Accuracy | 76,44 % | 76,19 % | 78,33 % | 78,24 % |
| Accuracy-Änderung | — | −0,25 Prozentpunkte | — | −0,09 Prozentpunkte |
| Keras-Parameter | 835.882 | 835.882 | 605.066 | 605.066 |
| Keras-Dateigröße | 10,31 MB | — | 7,81 MB | — |
| Float32-TFLite-Größe | 3,29 MB | — | 2,35 MB | — |
| INT8-TFLite-Größe | — | 988 KB | — | 847 KB |
| Float32-zu-INT8-TFLite-Verhältnis | — | 3,34× | — | 2,77× |

Interpretation: V2 ist in diesem Versuch genauer und besitzt weniger Keras-Parameter. INT8 erzeugt nur geringen Accuracy-Verlust und reduziert die TFLite-Datei deutlich. Die Modelle wurden mit maximal 30 Epochs, Early Stopping und Learning-Rate-Reduktion auf einer Kaggle-Tesla-P100 trainiert.

Keras und Netron können verschiedene Parameterzahlen anzeigen, weil BatchNormalization beim Export gefaltet wird und die Formate unterschiedlich zählen. Keras gegen TFLite ist außerdem kein vollständig formatgleicher Größenvergleich; Float32-TFLite gegen INT8-TFLite wäre strenger.

## 17. Bitbreitenanalyse

Ein Produkt aus zwei INT8-Werten kann mehr als 8 Bit benötigen. Danach werden viele Produkte addiert:

~~~text
Accumulator = Summe(Input × Weight)
~~~

Deshalb prüft man Produkt, Bias und Accumulator getrennt. Die typische Zielstruktur ist:

~~~text
Input: INT8
Weights: INT8
Bias: INT32
Accumulator: INT32
Output: INT8
~~~

Die finale Dokumentation sollte die maximale Summe und die dafür notwendige signed Bitbreite reproduzierbar berechnen und zeigen, dass sie höchstens 32 Bit benötigt.

## 18. Was du als Einschränkungen nennen solltest

1. Die ursprünglichen Werte stammen aus fünf Epochs; die aktuellen Werte stammen aus einem längeren Kaggle-Experiment.
2. Die korrigierte erste Convolution verwendet jetzt 16 Filter bei alpha=0.5; die bisherigen Kaggle-Ergebnisse stammen noch aus der vorherigen 32-Filter-Version.
3. Keras und Netron verwenden möglicherweise verschiedene Parameterzählungen.
4. Die bisherige Größenmessung nutzt unterschiedliche Dateiformate.
5. Die Bitbreitenanalyse ist jetzt als theoretische Worst-Case-Rechnung in results/bit_width_analysis.md dokumentiert.
6. Netron-Befunde sollten noch in results dokumentiert werden.
7. CIFAR-10 ist als Dataset erlaubt; die verwendete Dataset- und Trainingskonfiguration muss trotzdem exakt dokumentiert werden.

## 19. Typische Fragen

Warum MobileNet? Weil es für kleine, schnelle Edge- und Mobile-Modelle entwickelt wurde.

Warum V1 und V2? V2 ist eine Weiterentwicklung von V1; der Vergleich zeigt den Effekt von Bottlenecks und Residuals.

Warum Depthwise-Convolution? Sie filtert jeden Kanal separat und vermeidet viele Kanal-Kombinationen.

Warum 1×1-Convolution? Sie mischt Kanäle und verändert die Kanalanzahl.

Warum alpha=0.5? Für weniger Parameter, Speicher und Rechenaufwand.

Warum GlobalAveragePooling? Für einen kleinen Feature-Vektor ohne großen Dense-Layer.

Warum ReLU6? Für begrenzte Aktivierungen und gute Eignung für mobile/quantisierte Berechnung.

Warum keine ReLU nach der Projection in V2? Das ist der Linear Bottleneck; eine ReLU könnte Informationen verlieren.

Warum nur manchmal Add? Input und Output müssen dieselbe Form haben.

Warum INT32-Bias? Viele INT8-Produkte werden addiert, deshalb wird mehr Wertebereich benötigt.

Warum Representative Dataset? Sie liefert typische Aktivierungswerte für Scale- und Zero-Point-Kalibrierung.

Warum INT8-Accuracy-Verlust? Integer sind Näherungen von Float32.

Warum Netron? Es zeigt den tatsächlich exportierten Graphen und die tatsächlichen Tensor-Datentypen.

Warum kein FPGA? Die Aufgabe verlangt Analyse, aber ausdrücklich keine FPGA-Implementierung.

## 20. Präsentationsablauf

1. Ziel und feste Anforderungen.
2. Tensoren, Training, Loss und Accuracy.
3. CIFAR-10 und Vorverarbeitung.
4. Normale Convolution.
5. Depthwise-Separable-Convolution.
6. V1-Block und Shape-Verlauf.
7. V2-Block, Expansion, Projection und Skip.
8. Code in src/models.py.
9. Training und Speichern.
10. Float32-Ergebnisse.
11. Scale, Zero-Point und Representative Dataset.
12. Quantisierungscode.
13. INT8-Evaluation.
14. Netron-Befunde.
15. Vergleich von Accuracy, Parametern und Größe.
16. Einschränkungen und nächste Schritte.

## 21. Abschluss

> MobileNetV1 reduziert Rechenaufwand mit Depthwise-Separable-Convolutions. MobileNetV2 erweitert das Konzept um Inverted Residuals und Linear Bottlenecks. In meinem aktuellen Versuch ist V2 genauer und besitzt weniger Keras-Parameter. INT8 macht die Modelle deutlich kompakter, während die Accuracy nur geringfügig sinkt. Netron bestätigt INT8-Inputs, INT8-Gewichte, INT8-Aktivierungen und INT32-Bias-Tensoren. Die Werte sind vorläufig, weil nur fünf Epochs verwendet wurden und die exakte Anwendung von alpha auf die erste Convolution noch geprüft werden muss.
