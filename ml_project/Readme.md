**Инструкция по подготовке к запуску:**

Датасет требуется скачать с сайта https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci
После этого в директории configs требуется прописать новый путь до датасета для файлов **train_config.yml** и 
**eval_config.yml**. 

**Установка:**
```
python3 setup.py install
python3 setup.py build
```

**Запуск:**
```
python model/train.py --conf <path_to_train_conf.yml>
```

По умолчанию подтягивается файл из configs/train_config.yml

Информация по полям содержится в train_config.yml

**Валидация:**
```
python model/train.py --conf <path_to_eval_conf.yml>
```

По умолчанию подтягивается файл из configs/eval_config.yml

Информация по полям содержится в eval_config.yml

**Структура проекта:**

```bash

.
└── ml_project
    ├── build # build directory
    ├── configs # directory with configs
    │   ├── eval_config.yml #eval config
    │   └── train_config.yml #train config
    ├── dataset
    │   └── heart_cleveland_upload.csv #dataset
    ├── dist
    ├── __init__.py
    ├── logs #logs directory (will be created after first launch)
    │   ├── eval.log
    │   └── train.log
    ├── mlruns #directory with mlruns
    ├── model 
    │   ├── entities # 
    │   │   ├── custom_exception_class.py #custom class exceptions
    │   │   ├── feature_params.py #dataclass for features
    │   │   ├── __init__.py
    │   │   └── train_params.py #dataclass for train params
    │   ├── eval.py #evaluation script
    │   ├── __init__.py
    │   ├── modules #main code modules
    │   │   ├── __init__.py
    │   │   ├── metrics.py #metric calculation
    │   │   └── model.py #model module
    │   ├── preprocessing
    │   │   ├── data_load.py #data loading
    │   │   ├── __init__.py
    │   │   └── preprocessing.py #data preprocessing
    │   ├── trained
    │   │   └── model.pkl #example of saved model
    │   └── train.py #train script
    ├── notebooks #directory for notebooks
    │   └── EDA.ipynb #EDA notebook
    ├── Readme.md #Yeap, it's me
    ├── requirements.txt 
    ├── results #Directory with saved data
    │   ├── result_2022-11-06_21.23.25.385471.csv
    │   └── result_2022-11-06_21.24.24.491150.csv
    ├── setup.py
    └── tests #Directory with tests
        ├── __init__.py
        └── model_test.py
```
**Критерии (указаны максимальные баллы, по каждому критерию ревьюер может поставить баллы частично):**

- [X] В описании к пулл реквесту описаны основные "архитектурные" и тактические решения, которые сделаны в вашей работе. В общем, описание того, что именно вы сделали и для чего, чтобы вашим ревьюерам было легче понять ваш код (1 балл)
- [X] В пулл-реквесте проведена самооценка, распишите по каждому пункту выполнен ли критерий или нет и на сколько баллов(частично или полностью) (1 балл)

- [X] Выполнено EDA, закоммитьте ноутбук в папку с ноутбуками (1 балл)
   Вы так же можете построить в ноутбуке прототип(если это вписывается в ваш стиль работы)

- [ ] Можете использовать не ноутбук, а скрипт, который сгенерит отчет, закоммитьте и скрипт и отчет (за это + 1 балл)

- [X] Написана функция/класс для тренировки модели, вызов оформлен как утилита командной строки, записана в readme инструкцию по запуску (3 балла)
- [X] Написана функция/класс predict (вызов оформлен как утилита командной строки), которая примет на вход артефакт/ы от обучения, тестовую выборку (без меток) и запишет предикт по заданному пути, инструкция по вызову записана в readme (3 балла)

- [X] Проект имеет модульную структуру (2 балла)
- [X] Использованы логгеры (2 балла)

- [X] Написаны тесты на отдельные модули и на прогон обучения и predict (3 балла)

- [X] Для тестов генерируются синтетические данные, приближенные к реальным (2 балла)
   - можно посмотреть на библиотеки https://faker.readthedocs.io/, https://feature-forge.readthedocs.io/en/latest/
   - можно просто руками посоздавать данных, собственноручно написанными функциями.

- [X] Обучение модели конфигурируется с помощью конфигов в json или yaml, закоммитьте как минимум 2 корректные конфигурации, с помощью которых можно обучить модель (разные модели, стратегии split, preprocessing) (3 балла)
- [X] Используются датаклассы для сущностей из конфига, а не голые dict (2 балла)

- [X] Напишите кастомный трансформер и протестируйте его (3 балла)
   https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-the-step-by-step-guide-with-python-code-4a7d9b068156

- [X] В проекте зафиксированы все зависимости (1 балл)
- [X] Настроен CI для прогона тестов, линтера на основе github actions (3 балла).
Пример с пары: https://github.com/demo-ml-cicd/ml-python-package

PS: Можно использовать cookiecutter-data-science  https://drivendata.github.io/cookiecutter-data-science/ , но поудаляйте папки, в которые вы не вносили изменения, чтобы не затруднять ревью

Дополнительные баллы=)
- [ ] Используйте hydra для конфигурирования (https://hydra.cc/docs/intro/) - 3 балла

Mlflow
- [X] разверните локально mlflow или на какой-нибудь виртуалке (1 балл)
- [X] залогируйте метрики (1 балл)
- [ ] воспользуйтесь Model Registry для регистрации модели(1 балл)
  Приложите скриншот с вашим mlflow run
  DVC
- [ ] выделите в своем проекте несколько entrypoints в виде консольных утилит (1 балл).
  Пример: https://github.com/made-ml-in-prod-2021/ml_project_example/blob/main/setup.py#L16
  Но если у вас нет пакета, то можно и просто несколько скриптов

- [ ] добавьте датасет под контроль версий (1 балл)
  
- [ ] сделайте dvc пайплайн(связывающий запуск нескольких entrypoints) для изготовления модели(1 балл)

Для большего удовольствия в выполнении этих частей рекомендуется попробовать подключить удаленное S3 хранилище(например в Yandex Cloud, VK Cloud Solutions или Selectel)

