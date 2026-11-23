import glob
import json
import os


class CompileCommandMerger:
    OUTPUT_NAME = 'compile_commands.json'

    def merge(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output = os.path.join(root, 'build', self.OUTPUT_NAME)
        entries = []
        pattern = os.path.join(root, 'build', '*', self.OUTPUT_NAME)
        for path in sorted(glob.glob(pattern)):
            with open(path, 'r', encoding='utf-8') as stream:
                entries.extend(json.load(stream))
        entries = [entry for entry in entries if os.path.exists(entry.get('file', ''))]
        with open(output, 'w', encoding='utf-8') as stream:
            json.dump(entries, stream, indent=2)
        print('Merged %d compile commands into %s' % (len(entries), output))


if __name__ == '__main__':
    CompileCommandMerger().merge()
