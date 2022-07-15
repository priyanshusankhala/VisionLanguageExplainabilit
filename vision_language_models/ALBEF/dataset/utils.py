import re
from typing import List, Union


def clean_caption(caption: Union[str, List[str]], max_words=30) -> List:
    if isinstance(caption, str):
        caption = [caption]

    def _clean(s: str) -> str:
        s = re.sub(
            r"([,.'!?\"()*#:;~])",
            '',
            s.lower(),
        ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person').replace('_', ' ')

        s = re.sub(
            r"\s{2,}",
            ' ',
            s,
        )
        s = s.rstrip('\n')
        s = s.strip(' ')

        # truncate caption
        caption_words = s.split(' ')
        if len(caption_words) > max_words:
            s = ' '.join(caption_words[:max_words])
        return s

    return list(map(_clean, caption))
