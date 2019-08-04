"use strict";
// ID3
// =========================
// CLASSES
// =========================
class AttributeValuePair {
    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
}
class Sample {
    constructor(values, result) {
        this.values = values;
        this.result = result;
    }
    get_value(attr_name) {
        var value_pair = this.values.find((v) => v.name == attr_name);
        return (value_pair) ? value_pair.value : '';
    }
}
class Attribute {
    constructor(name, values) {
        this.name = name;
        this.values = values;
    }
}
class TreeNodeChild {
    constructor(node, value) {
        this.node = node;
        this.value = value;
    }
}
class Rule {
    constructor(pairs, result) {
        this.pairs = pairs;
        this.result = result;
    }
}
class TreeNode {
    constructor(children) {
        this.children = children;
    }
    classify(sample) {
        if (!this.attribute) {
            //Is leaf
            return this.label;
        }
        let attribute_value = sample.get_value(this.attribute.name);
        var next_node_result = this.children.find((child) => child.value == attribute_value);
        if (!next_node_result) {
            console.warn("This should not be happening.");
            return false;
        }
        let next_node = next_node_result;
        return next_node.node.classify(sample);
    }
    get_rules() {
        if (!this.attribute) {
            //Is leaf
            return [];
        }
        let rules = [];
        this.children.forEach((child) => {
            let pair = new AttributeValuePair(this.attribute.name, child.value);
            let result;
            if (!child.node.attribute) {
                result = child.node.label;
                rules.push(new Rule([pair], result));
            }
            else {
                let child_rules = child.node.get_rules();
                child_rules.map((rule) => rule.pairs.push(pair));
                rules = rules.concat(child_rules);
            }
        });
        return rules;
    }
}
// =========================
// HELPERS
// =========================
let file_content = '';
// =========================
// HELPERS
// =========================
function print_result(output) {
    let result = document.getElementById("result");
    if (result != null) {
        result.innerHTML = output;
        return true;
    }
    return false;
}
function read_single_file(e) {
    var file = e.target.files[0];
    if (!file) {
        return;
    }
    var reader = new FileReader();
    reader.onload = function (e) {
        var contents = e.target.result;
        file_content = contents;
        main();
    };
    reader.readAsText(file);
}
function generalOnLoad() {
    document.getElementById('file-input').addEventListener('change', read_single_file, false);
}
// =========================
// PROCEDURES
// =========================
function get_positive_negative(samples) {
    let positive_samples = samples.filter((sample) => sample.result);
    let negative_samples = samples.filter((sample) => !sample.result);
    return [positive_samples, negative_samples];
}
function entropy(samples) {
    var pos_neg = get_positive_negative(samples);
    let positive_samples = pos_neg[0];
    let negative_samples = pos_neg[1];
    let n_total = samples.length;
    let n_positive = positive_samples.length;
    let n_negative = negative_samples.length;
    let p_positive = n_positive / n_total;
    let p_negative = n_negative / n_total;
    //To avoid log(0)
    p_positive += 0.0001;
    p_negative += 0.0001;
    return (-p_positive * Math.log2(p_positive) - p_negative * Math.log2(p_negative));
}
function get_best_attribute(samples, attributes) {
    let original_entropy = entropy(samples);
    let max_information_gain = 0;
    let best_attribute = null;
    attributes.forEach((attribute) => {
        let information_gain = original_entropy;
        attribute.values.forEach((attr_value) => {
            let classified_samples = samples.filter((sample) => sample.get_value(attribute.name) == attr_value);
            let proportion = classified_samples.length / samples.length;
            if (classified_samples.length > 0) {
                information_gain -= proportion * entropy(classified_samples);
            }
        });
        if (information_gain > max_information_gain) {
            max_information_gain = information_gain;
            best_attribute = attribute;
        }
    });
    return best_attribute;
}
function ID3(samples, possible_attributes) {
    let root = new TreeNode([]);
    var pos_neg = get_positive_negative(samples);
    let positive_samples = pos_neg[0];
    let negative_samples = pos_neg[1];
    if (positive_samples.length == 0 || negative_samples.length == 0) {
        if (positive_samples.length > 0) {
            root.label = true;
        }
        else if (negative_samples.length > 0) {
            root.label = false;
        }
        else {
            //No samples
            root.label = (positive_samples.length > negative_samples.length) ? true : false;
            // console.log("  [+, -]: [" + positive_samples.length + ", " + negative_samples.length + "]");
        }
        return root;
    }
    if (possible_attributes.length == 0) {
        root.label = (positive_samples.length > negative_samples.length) ? true : false;
        // console.log("  [+, -]: [" + positive_samples.length + ", " + negative_samples.length + "]");
        return root;
    }
    var best_attribute_result = get_best_attribute(samples, possible_attributes);
    if (best_attribute_result == null) {
        console.warn("This should not be happening.");
        return root;
    }
    let best_attribute = best_attribute_result;
    root.attribute = best_attribute;
    let new_possible_attributes = possible_attributes.filter((attr) => best_attribute && attr.name != best_attribute.name);
    best_attribute.values.forEach((attr_value) => {
        let classified_samples = samples.filter((sample) => sample.get_value(best_attribute.name) == attr_value);
        var pos_neg = get_positive_negative(classified_samples);
        let positive_samples = pos_neg[0];
        let negative_samples = pos_neg[1];
        let new_node = ID3(classified_samples, new_possible_attributes);
        root.children.push(new TreeNodeChild(new_node, attr_value));
    });
    return root;
}
function main() {
    let attributes = [
        new Attribute('Embarked', ['C', 'Q', 'S']),
        new Attribute('Pclass', ['1', '2', '3']),
        new Attribute('Sex', ['male', 'female']),
    ];
    // let attributes: Attribute[] = [
    //     new Attribute('chuva',    ['nenhuma',        'muita']),
    //     new Attribute('horario',      ['manha', 'tarde']),
    // ];
    let samples = [];
    let header = file_content.split("\n")[0].replace("\r", "").split(";");
    let lines = file_content.split("\n");
    for (var i = 1; i < lines.length; i++) {
        let values = lines[i].replace("\r", "").split(";");
        let result = false;
        let parsed = [];
        for (var j = 0; j < values.length; j++) {
            switch (header[j]) {
                case 'Survived':
                    result = (values[j] === '1');
                    break;
                default:
                    let corresponding_attr = attributes.find((f) => f.name == header[j]);
                    if (corresponding_attr && corresponding_attr.values.find((f) => f == values[j])) {
                        parsed.push(new AttributeValuePair(header[j], values[j]));
                    }
                    break;
            }
        }
        samples.push(new Sample(parsed, result));
    }
    //k-folding
    let k = 10;
    let fold_size = Math.trunc(samples.length / k);
    let correct_class = 0;
    let total_tested = 0;
    for (var i = 0; i < k; i++) {
        let train_samples_1 = samples.slice(0, i * fold_size);
        let validate_samples = samples.slice(i * fold_size, (i + 1) * fold_size);
        let train_samples_2 = samples.slice((i + 1) * fold_size, samples.length);
        let train_samples = train_samples_1.concat(train_samples_2);
        let tree = ID3(train_samples, attributes);
        let rules = tree.get_rules();
        let fold_correct_class = 0;
        if (tree != null) {
            validate_samples.forEach((sample) => {
                total_tested++;
                let appropriate_rule = rules.find((rule) => rule.pairs.every((pair) => sample.get_value(pair.name) == pair.value));
                if (!appropriate_rule) {
                    console.warn("No rule was found.");
                    return;
                }
                // if (tree.classify(sample) == sample.result) {
                if (appropriate_rule.result == sample.result) {
                    fold_correct_class++;
                }
            });
        }
        // console.log("Fold #" + i + ": " + (100 * fold_correct_class / validate_samples.length) + "%")
        correct_class += fold_correct_class;
    }
    console.log("Prediction rate: " + (100 * correct_class / total_tested) + "%");
}
//# sourceMappingURL=index.js.map