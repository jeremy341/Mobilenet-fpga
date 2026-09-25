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

Eine Conv2D mit Kernel 3×3, Stride 2 und `same`-Padding halbiert ungefähr die räumliche Auflösung. In diesen Modellen wird der Stem mit `int(32 * alpha)` Filtern gebaut; bei alpha=0.5 sind das 16 Kanäle.

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

Der Input wird durch die initiale Convolution von 128×128×3 auf 64×64×16 gebracht. Danach folgen 13 Depthwise-Separable-Blöcke.

Die Liste blocks enthält Tupel:

~~~text
(Outputkanäle vor alpha, Stride, Wiederholungen)
~~~

Die Schleife multipliziert die Outputkanäle mit alpha, erzeugt die Wiederholungen und setzt nach dem ersten Block einer Stage den Stride auf 1. So downsamplet nur der erste Block einer wiederholten Stage.

Der räumliche Verlauf ist:

~~~text
128×128×3
→ Stem: 64×64×16
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
→ Stem: 64×64×16
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

Die initiale Conv2D wird ebenfalls mit alpha multipliziert. Bei alpha=0.5 besitzt sie 16 Filter. Die korrigierten Kaggle-Artefakte und die Standarddateien im Repository stammen aus dieser Architektur.

## 11. Training

train_v1.py und train_v2.py importieren Path, die Datenfunktionen und den jeweiligen Model Builder.

ROOT findet mit Path(__file__).resolve().parent.parent das Projektverzeichnis. MODELS_DIR zeigt auf models.

load_data lädt die Splits. make_dataset erzeugt Training und Validation. Die Testdaten werden beim Training bewusst nicht verwendet.

build_mobilenet_v1 beziehungsweise build_mobilenet_v2 erzeugt das Modell. model.fit trainiert maximal 30 Epochs und prüft nach jeder Epoch auf Validation. EarlyStopping überwacht val_accuracy mit patience 7 und stellt die besten Weights wieder her. ReduceLROnPlateau überwacht val_loss, reduziert die Learning Rate um den Faktor 0.2 und verwendet patience 3 sowie eine minimale Learning Rate von 1e-6.

MODELS_DIR.mkdir legt den Speicherordner an. model.save schreibt die Float32-Datei mobilenet_v1.keras beziehungsweise mobilenet_v2.keras.

Beide Modelle verwenden dieselbe Konfiguration. Seed 67 wird für Python, NumPy und TensorFlow gesetzt.

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

LiteRT quantisiert Conv-Gewichte symmetrisch in `[-127,127]`, während Aktivierungen `[-128,127]` nutzen. Daher ist ein konservativer maximaler Produktbetrag `128×127=16256`. Viele solcher Produkte werden addiert; die konkrete Bitbreitenrechnung ist in `results/bit_width_analysis.md` dokumentiert.

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

Die korrigierten Dateien wurden in Netron geprüft. Beide zeigen:

~~~text
Input:  int8[-1,128,128,3]
Output: int8[-1,10]
Stem-Filter: int8[16,3,3,3]
Stem-Bias:   int32[16]
~~~

Damit bestätigt der Graph die alpha-scalierte 16-Filter-Stem-Convolution.

V1 zeigt eine initiale Conv2D, danach DepthwiseConv2D/Pointwise-Conv2D-Paare, anschließend Mean, FullyConnected und Softmax. Die finale Classifier-Matrix ist `int8[10,512]`, der Bias `int32[10]`. Netron meldet 818.252 Inference-Graph-Parameter.

V2 zeigt Expansion-Conv2D, DepthwiseConv2D, lineare Projection-Conv2D und zehn Add-Nodes für Residual Connections. Die Expansion erreicht je nach Stage 48, 72, 96, 192, 288 oder 480 Kanäle; die finale Feature-Convolution besitzt 640 Kanäle. Die Classifier-Matrix ist `int8[10,640]`, der Bias `int32[10]`. Netron meldet 578.652 Inference-Graph-Parameter.

Beide Modelle enthalten INT8- und INT32-Tensoren. Die Graph Properties zeigen:

~~~text
Input:  0.003921568859368563 × (q + 128)
Output: 0.00390625 × (q + 128)
~~~

Das entspricht Input scale `1/255`, Input zero-point `-128`, Output scale `1/256` und Output zero-point `-128`. Die Interpreter-Zählung lautet für V1 59 INT8/29 INT32-Tensoren und für V2 119 INT8/54 INT32-Tensoren. Ausführliche Shapes und Eigenschaften stehen in `results/netron_inspection.md`.

Keras und Netron Parameterzahlen sind verschieden: Keras zählt das trainierbare Modell, Netron das konvertierte Inference-Graph. Die Conversions falten BatchNormalization und Konstanten in die Inference-Operationen. Für den Architekturvergleich verwende ich durchgehend die Keras-Parameterzahlen.

## 16. Ergebnisse

Die bisher dokumentierten Werte:

| Messgröße | V1 Float32 | V1 INT8 | V2 Float32 | V2 INT8 |
|---|---:|---:|---:|---:|
| Test Accuracy | 76,72 % | 76,35 % | 77,93 % | 77,94 % |
| Accuracy-Änderung | — | −0,37 Prozentpunkte | — | +0,01 Prozentpunkte |
| Keras-Parameter | 834.666 | 834.666 | 604.234 | 604.234 |
| Keras-Dateigröße | 10,30 MB | — | 7,80 MB | — |
| Float32-TFLite-Größe | 3,29 MB | — | 2,35 MB | — |
| INT8-TFLite-Größe | — | 986 KB | — | 845 KB |
| Float32-zu-INT8-TFLite-Reduktion | — | 70,04 % (3,34×) | — | 63,97 % (2,78×) |

Interpretation: V2 ist in diesem Versuch genauer und besitzt 27,61 % weniger Keras-Parameter. V1 verliert durch INT8 0,37 Prozentpunkte. V2 liegt im Test um 0,01 Punkte höher als Float32; das entspricht genau einer zusätzlichen richtigen Vorhersage von 10.000 Testbildern. Beide INT8-Dateien sind deutlich kleiner als ihre Float32-TFLite-Versionen. Trainiert wurde auf einer Kaggle-Tesla-P100 mit maximal 30 Epochs, Early Stopping und Learning-Rate-Reduktion.

Keras zählt 834.666 V1- und 604.234 V2-Parameter. Netron meldet 818.252 beziehungsweise 578.652 Parameter im exportierten Inference-Graph. Die Größenmessung vergleicht hier formatgleiche Float32-TFLite- und INT8-TFLite-Dateien.

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

Die Worst-Case-Rechnung steht in `results/bit_width_analysis.md`. Sie verwendet die LiteRT-Ranges für INT8-Aktivierungen und symmetrische INT8-Gewichte. Die größte betrachtete Summe ist 10.403.840 und benötigt 25 signed Bits; Bias-Tensoren sind INT32.

## 18. Was du als Einschränkungen nennen solltest

1. Die Trainingshistorien zeigen Overfitting: Trainingsgenauigkeit nähert sich 100 %, während die Validation früher stagniert.
2. Keras und Netron zählen Parameter in unterschiedlichen Modellrepräsentationen.
3. Die Kaggle-Ausgabe enthält keine Python-/TensorFlow-Versionsmetadaten; diese Laufzeitversionen daher nicht behaupten.
4. Die Bitbreitenrechnung ist eine theoretische Worst-Case-Abschätzung nach der LiteRT-Gewichts- und Aktivierungsrange.

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

> MobileNetV1 reduziert Rechenaufwand mit Depthwise-Separable-Convolutions. MobileNetV2 ergänzt Inverted Residuals und Linear Bottlenecks. Im korrigierten alpha=0.5-Lauf erreicht V2 77,93 % Float32- und 77,94 % INT8-Testgenauigkeit bei 604.234 Keras-Parametern; V1 erreicht 76,72 % und 76,35 % bei 834.666 Parametern. Netron bestätigt für beide den 16-Filter-Stem sowie INT8-Inputs/Gewichte/Aktivierungen und INT32-Bias-Tensoren. Die INT8-Dateien sind 70,04 % (V1) beziehungsweise 63,97 % (V2) kleiner als Float32-TFLite. Die Trainingskurven zeigen Overfitting, das ich bei der Interpretation berücksichtige.
