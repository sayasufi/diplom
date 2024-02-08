<a href="https://wakatime.com/badge/user/018c3f04-b140-41f9-a489-5b0143d153f5/project/018c3f07-b236-41ed-9da1-1473dedaae40"><img src="https://wakatime.com/badge/user/018c3f04-b140-41f9-a489-5b0143d153f5/project/018c3f07-b236-41ed-9da1-1473dedaae40.svg" alt="wakatime"></a>

# Алгоритм БИНС и РТСЛН БПЛА

## Используемые фреймворки и библиотеки
<ul>
<li><strong>Python</strong></li>
<li><strong>Numpy</strong></li>
<li><strong>Scipy</strong></li>
<li><strong>Folium</strong></li>
<li><strong>Matplotlib</strong></li>
<li><strong>Pandas</strong></li>
<li><strong>Seaborn</strong></li>
<li><strong>Pyins</strong></li>
</ul>

## Функционал

#### 1) Алгоритм БИНС для наземного транспорта и БПЛА, вычисление ошибок ориентации, координат и скоростей
<img src="bins/png/2/2.png">

#### 2) Алгоритм РТСЛН и коррекция по РТСЛН, СНС, магнитометру, одометру, барометру
<img src="bins/график.png">

#### 3) Построение тепловой карты ошибок определения координат по маякам (математическое ожидание + )
<img src="bins/png/reger.png">

#### 4) Построение траектории полета на реальной карте
<img src="bins/png/semyon/tr.png">

#### 5) Перевод lan / lon (4326) в сферические координаты (3857) и обратно

#### 6) Фильтр Калмана и метод скользящего среднего