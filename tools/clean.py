import re


def clean(text):
    text = ' '.join(text.split())
    text = text.replace("\u3002", ".")
    text = text.replace("\uFF1F", "?")
    text = text.replace("\uFF01", "!")
    text = text.replace("\uFF0C", ",")
    text = text.replace("\u3001", ",")
    text = text.replace("\uFF1B", ";")
    text = text.replace("\uFF1A", ":")
    text = re.sub(r"[\u300C\u300D\u300E\u300F\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201C\u201D\u300A\u300B\u3008\u3009]", "\"", text)
    text = re.sub(r"[\uFF08\u3014\u3010]", "(", text)
    text = re.sub(r"[\uFF09\u3015\u3011]", ")", text)
    text = text.replace("\u2014", "-")
    text = text.replace("\u2026", "...")
    text = text.replace("\u2013", "-")
    text = text.replace("\uFF0E", ",")
    text = re.sub(r"\.+", '.', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text
