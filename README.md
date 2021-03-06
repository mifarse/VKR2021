# Выявление паттернов функционирования датчиков для диагностики состояния здания
**Автор**: Уруков Серафим Дмитриевич  
**Руководитель ВКР**: Новикова Евгения Сергеевна, к.т.н., доцент кафедры ИС

Предметом работы является набор пакетных программ и модулей, реализованных на языке программирования Python, которые упростят процесс мониторинга данных в информационной системе автоматизированных зданий и позволят контролировать состояние здания, на основе больших данных.

**Цель**: в рамках выполнения задания выпускной квалификационной работы необходимо исследовать принципы хранения time-series данных, получаемых от контроллеров, обработать большие данные, выявить паттерны поведения показателей температур.

Разработанный подход использует средства визуального анализа данных. Идея такова: Состояние системы в некоторый момент времени представимо в виде точки многомерного пространство, само функционирование системы может быть представлено в виде траектории в многомерном пространстве. Ранее было показано, что проекция точек в двумерное пространство формирует графические паттерны, характерные для разных состояний системы. Поэтому предлагается использовать методы декомпозиции данных (PCA) и триангуляции Делоне для того, чтобы охарактеризовать временные данные на заданном интервале. 


Содержание: для достижения цели была изучена документация следующих программных продуктов:
- математических пакетов NumPy, pandas, SciPy, scikit;
- облачной инфраструктуры Yandex Cloud;
- интерактивной IDE для математических расчетов JupyterLab.

По итогам изучения был разработан метод, позволяющий выявлять проблемы в показаниях сенсоров на основе построенных паттернов поведения данных.


## Сервер, на котором запускались расчеты
| Конфигурация сервера         |                        |
|------------------------------|------------------------|
| Инфраструктура               | Yandex Cloud           |
| Зона доступности             | ru-central1-c          |
| Платформа                    | **Intel Cascade Lake** |
| Гарантированная доля vCPU    | **50%**                |
| vCPU                         | **2**                  |
| RAM                          | **4 Гб**               |
| Объём дискового пространства | **32 ГБ**              |

## Результаты работы

- Изучены методы сокращения размерности, автоматического аннотирования данных, построения триангуляции по множеству точек; опробованы современные подходы разработки программного обеспечения – облачная инфраструктура, система контроля версий, параллельные вычисления, кэширование расчетов.
- Разработан подход, который может оценить состояние системы на основе многомерных данных. Эта оценка действительно отражает состояние системы.
- На основе получаемой оценки можно сделать предположение о том, находится ли система в нормальном состоянии.
- Автоматическое аннотирование в рамках данного подхода позволяет упростить разметку данных за счет выявления паттернов.
- Предложенный подход работает на больших данных.
- Разработан программный python-модуль по выполнению расчетов. Дальнейшая работа может быть связана с созданием полноценного приложения с панелью мониторинга для анализа многомерных данных и диагностики состояния системы.
