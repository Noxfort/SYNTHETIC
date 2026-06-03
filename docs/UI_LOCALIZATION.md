# UI Localization (i18n)

The `SYNTHETIC` graphical interface is fully internationalized (i18n), supporting dynamic hot-swapping of languages without requiring application restarts. 

## 1. The Locale Dictionaries

All hardcoded UI strings are abstracted into JSON files located in `ui/locale/`. 
Current supported languages include:
- English (`en.json`) - *Default*
- Portuguese - Brazil (`pt-br.json`)
- French (`fr.json`)
- Spanish (`es.json`)
- Russian (`ru.json`)
- Mandarin - Simplified (`zh-cn.json`)

To add a new language, developers simply need to create a new JSON dictionary mapping the standard keys to translated values, and register the short-code in `translator.py`.

## 2. The Translator Singleton

The `Translator` (`ui/translator.py`) operates as a Singleton class. This guarantees that regardless of which UI window or pop-up requests a string, they are all querying the exact same state and language file.

### Usage
Components import the global `translator` instance:
```python
from ui.translator import translator
text = translator.t("map_title")
```

The translator also supports Python-style string formatting for dynamic values (e.g., passing the number of cameras remaining to be placed).

## 3. Dynamic Hot-Swapping

In `ui/gui.py`, a `ttk.Combobox` is bound to the `<<ComboboxSelected>>` event. When a user selects a new language:
1. The `Translator` singleton loads the new JSON file into memory.
2. The `update_ui_texts()` method is triggered across the application.
3. Every `ttk.Label`, `ttk.Button`, `ttk.LabelFrame`, and `ttk.Checkbutton` has its `.config(text=...)` property overwritten with the new translation.
4. Active sub-windows (like `MapSelectorWindow`) instantly reflect the new language via their local `update_instructions()` polling logic.
