# Задача 6 — один шаг линейной регрессии

Исходный код: [`task6.py`](task6.py)

## Что сделано

Для модели `y_hat = wx+b` вручную реализованы MSE, аналитические градиенты по `w` и `b`, численная проверка градиентов и один шаг градиентного спуска. После шага loss уменьшается с `15.0` примерно до `0.2341`.

## Фокус обучения

- линейная регрессия;
- MSE;
- аналитические производные;
- численный gradient checking;
- learning rate;
- обновление параметров против градиента.

## Формулы

```text
y_hat[i] = w·x[i] + b
L = (1/N) Σ(y_hat[i] - y[i])²
dL/dw = (2/N) Σ(y_hat[i] - y[i])x[i]
dL/db = (2/N) Σ(y_hat[i] - y[i])
```

Шаг обучения:

```text
w_new = w - learning_rate · dL/dw
b_new = b - learning_rate · dL/db
```

## Код построчно

```python
import numpy as np
```

Подключает NumPy для массивов.

```python
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 5.0])
```

Создаёт три входа и три целевых значения.

```python
w = 0.0
b = 0
```

Задаёт начальные вес и смещение, равные нулю. Поэтому начальные предсказания также нулевые.

```python
a = 0.1
```

Задаёт скорость обучения. Имя `a` здесь означает learning rate; более понятное имя — `eta` или `learning_rate`.

```python
def predict(x,w,b):
    return x * w + b
```

Объявляет линейную модель и сразу применяет её ко всему массиву `x`.

```python
def mse(yhat, y):
```

Объявляет MSE, вычисляемую циклом.

```python
    sum = 0
```

Создаёт сумму квадратов ошибок. Имя `sum` перекрывает встроенную функцию Python; `total` было бы безопаснее.

```python
    for i in range(len(y)):
```

Перебирает обучающие объекты.

```python
        sum +=(yhat[i]-y[i])**2
```

Добавляет квадрат ошибки объекта.

```python
    return sum/len(yhat)
```

Усредняет сумму и возвращает loss.

```python
def gradient_dw(yhat, x,y):
```

Объявляет аналитическую производную loss по весу.

```python
    sum = 0
```

Создаёт аккумулятор суммы.

```python
    for i in range(len(y)):
```

Перебирает все объекты.

```python
        sum +=(yhat[i]-y[i])*x[i]
```

Добавляет произведение ошибки на вход. Множитель `x[i]` появляется потому, что `∂(wx+b)/∂w = x`.

```python
    return 2*sum/len(yhat)
```

Умножает на `2/N` согласно производной MSE и возвращает `dL/dw`.

```python
def gradient_db(yhat, x,y):
```

Объявляет аналитическую производную по смещению. Аргумент `x` здесь не используется и мог бы быть удалён.

```python
    sum = 0
```

Создаёт аккумулятор.

```python
    for i in range(len(y)):
```

Перебирает объекты.

```python
        sum +=(yhat[i]-y[i])
```

Суммирует ошибки. Дополнительного `x[i]` нет, потому что `∂(wx+b)/∂b = 1`.

```python
    return 2*sum/len(yhat)
```

Возвращает `dL/db`.

```python
def num_grad(w,x,y,b,h=1e-5):
```

Объявляет численную проверку градиентов.

```python
    w_plus = predict(x,w+h,b)
    w_minus = predict(x,w-h,b)
```

Получает предсказания при весе, увеличенном и уменьшенном на `h`.

```python
    b_plus = predict(x,w,b+h)
    b_minus = predict(x,w,b-h)
```

Получает предсказания при двух сдвинутых значениях `b`.

```python
    grad_num_dw = (mse(w_plus,y)-mse(w_minus,y))/(2*h)
```

Вычисляет численную производную loss по `w` центральной разностью.

```python
    grad_num_db = (mse(b_plus,y)-mse(b_minus,y))/(2*h)
```

Вычисляет численную производную по `b`.

```python
    return grad_num_dw, grad_num_db
```

Возвращает обе производные.

```python
yhat = predict(x,w,b)
```

Получает начальные предсказания `[0, 0, 0]`.

```python
loss_before = mse(yhat, y)
```

Считает начальный loss, равный `15.0`.

```python
dl_dw = gradient_dw(yhat,x,y)
dl_db = gradient_db(yhat,x,y)
```

Получает аналитические градиенты по обоим параметрам.

```python
grad_num_dw, grad_num_db = num_grad(w,x,y,b,h=1e-5)
```

Получает численные градиенты для проверки формул.

```python
w_new = w- a*dl_dw
b_new = b-a*dl_db
```

Делает один шаг градиентного спуска. Градиенты отрицательны, поэтому новые `w` и `b` становятся больше нуля.

```python
yhat_new = predict(x, w_new, b_new)
```

Вычисляет предсказания после обновления.

```python
loss_after = mse(yhat_new, y)
```

Считает новый loss.

```python
print("loss before:", loss_before)
print("loss after:", loss_after)
```

Показывает, что ошибка уменьшилась с `15.0` примерно до `0.2341`.

```python
print("analytical gradient dl_dw:", dl_dw)
print("analytical gradient dl_db:", dl_db)
```

Печатает аналитические производные: примерно `-16.6667` и `-7.3333`.

```python
print("numerical gradient dw:", grad_num_dw)
print("numerical gradient db:", grad_num_db)
```

Печатает численные производные; они почти полностью совпадают с аналитическими.

```python
print("compare after and before loss:", loss_before > loss_after)
```

Печатает `True`, если один шаг действительно уменьшил loss. В отличие от предыдущих задач, здесь это только вывод, а не `assert`, поэтому программа не остановится при неудачном обновлении.

## Ожидаемый результат

```text
loss before: 15.0
loss after:  около 0.2341
dL/dw:       около -16.6667
dL/db:       около -7.3333
```

## Вопросы для повторения

1. Почему в `dL/dw` присутствует `x[i]`, а в `dL/db` нет?
2. Почему параметры обновляются со знаком минус?
3. Что означает отрицательный градиент в начальной точке?
4. Как проверить, что формулы градиентов написаны правильно?
5. Что может произойти при слишком большом learning rate?

