// ID3

// =========================
// CLASSES
// =========================

class AttributeValuePair {
    name: string;
    value: string;

    constructor(name: string, value: string) {
        this.name = name;
        this.value = value;
    }
}

class Sample {
    values: AttributeValuePair[];
    result: boolean;

    constructor(values: AttributeValuePair[], result: boolean) {
        this.values = values;
        this.result = result;
    }

    public get_value(attr_name: string):string {
        var value_pair = this.values.find((v) => v.name == attr_name);
        return (value_pair) ? value_pair.value : '';
    }
}

class Attribute {
    name: string;
    values: string[];

    constructor(name: string, values: string[]) {
        this.name = name;
        this.values = values;
    }
}

class TreeNodeChild {
    node: TreeNode;
    value: string;

    constructor(node: TreeNode, value: string) {
        this.node = node;
        this.value = value;
    }
}

class Rule {
    pairs: AttributeValuePair[];
    result: boolean;

    constructor(pairs: AttributeValuePair[], result: boolean) {
        this.pairs = pairs;
        this.result = result;
    }
}

class TreeNode {
    children: TreeNodeChild[];
    attribute?: Attribute;
    label?: boolean;

    constructor(children: TreeNodeChild[]) {
        this.children = children;
    }

    public classify(sample: Sample):any {
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
        
        let next_node: TreeNodeChild = next_node_result;
        return next_node.node.classify(sample);
    }

    public get_rules(): Rule[] {
        if (!this.attribute) {
            //Is leaf
            return [];
        }

        let rules: Rule[] = [];

        this.children.forEach((child) => {
            let pair: AttributeValuePair = new AttributeValuePair(this.attribute!.name, child.value);
            let result: boolean;

            if (!child.node.attribute) {
                result = child.node.label!;
                rules.push(new Rule([pair], result));
            }
            else {
                let child_rules: Rule[] = child.node.get_rules();
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

let file_content: string = '';

// =========================
// HELPERS
// =========================

function print_result(output: string) {
    let result = document.getElementById("result");
    if (result != null) {
        result.innerHTML = output;
        return true;
    }
    return false;
}

function read_single_file(e: any):void {
    var file = e.target.files[0];
    if (!file) { return; }
    var reader = new FileReader();
    reader.onload = function(e) {
        var contents = e.target!.result;
        file_content = contents;
        main();
    };
    reader.readAsText(file);
}

function generalOnLoad() {
    document.getElementById('file-input')!.addEventListener('change', read_single_file, false);
}

// =========================
// PROCEDURES
// =========================

function get_positive_negative(samples: Sample[]):Sample[][] {
    let positive_samples: Sample[] = samples.filter((sample) => sample.result);
    let negative_samples: Sample[] = samples.filter((sample) => !sample.result);
    return [positive_samples, negative_samples];
}

function entropy(samples: Sample[]):number {
    var pos_neg = get_positive_negative(samples);
    let positive_samples: Sample[] = pos_neg[0];
    let negative_samples: Sample[] = pos_neg[1];

    let n_total: number = samples.length;
    let n_positive: number = positive_samples.length;
    let n_negative: number = negative_samples.length;

    let p_positive: number = n_positive / n_total;
    let p_negative: number = n_negative / n_total;

    //To avoid log(0)
    p_positive += 0.0001;
    p_negative += 0.0001;

    return (- p_positive * Math.log2(p_positive) - p_negative * Math.log2(p_negative));
}

function get_best_attribute(samples: Sample[], attributes: Attribute[]):Attribute | null {
    let original_entropy: number = entropy(samples);
    let max_information_gain: number = 0;
    let best_attribute: Attribute | null = null;
    
    attributes.forEach((attribute) => {
        let information_gain: number = original_entropy;

        attribute.values.forEach((attr_value) => {
            let classified_samples: Sample[] = samples.filter((sample) => sample.get_value(attribute.name) == attr_value);
            let proportion: number = classified_samples.length / samples.length;

            if (classified_samples.length > 0) {
                information_gain -= proportion * entropy(classified_samples);
            }
        });
        
        if (information_gain > max_information_gain) {
            max_information_gain = information_gain;
            best_attribute = attribute;
        }
    })
    
    return best_attribute;
}

function ID3(samples: Sample[], possible_attributes: Attribute[]):TreeNode {
    let root = new TreeNode([]);

    var pos_neg = get_positive_negative(samples);
    let positive_samples: Sample[] = pos_neg[0];
    let negative_samples: Sample[] = pos_neg[1];

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
    let best_attribute: Attribute = best_attribute_result; 
    root.attribute = best_attribute;

    let new_possible_attributes: Attribute[] =
    possible_attributes.filter((attr) => best_attribute && attr.name != best_attribute.name);

    best_attribute.values.forEach((attr_value) => {
        let classified_samples: Sample[] = 
            samples.filter((sample) => sample.get_value(best_attribute.name) == attr_value);

        var pos_neg = get_positive_negative(classified_samples);
        let positive_samples: Sample[] = pos_neg[0];
        let negative_samples: Sample[] = pos_neg[1];
        
        let new_node: TreeNode = ID3(classified_samples, new_possible_attributes);
        root.children.push(new TreeNodeChild(new_node, attr_value));
    });

    return root;
}

function main() {
    let attributes: Attribute[] = [
        new Attribute('Embarked',    ['C',    'Q',     'S']),
        new Attribute('Pclass',      ['1',    '2',     '3']),
        new Attribute('Sex',         ['male', 'female'    ]),
    ];
    
    // let attributes: Attribute[] = [
    //     new Attribute('chuva',    ['nenhuma',        'muita']),
    //     new Attribute('horario',      ['manha', 'tarde']),
    // ];

    let samples: Sample[] = [];

    let header: string[] = file_content.split("\n")[0].replace("\r", "").split(";");
    let lines: string[] = file_content.split("\n");

    for (var i = 1; i < lines.length; i++) {
        let values: string[] = lines[i].replace("\r", "").split(";");
        
        let result: boolean = false;
        let parsed: AttributeValuePair[] = [];

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
    let fold_size: number = Math.trunc(samples.length / k);

    let correct_class: number = 0;
    let total_tested: number = 0;

    for (var i = 0; i < k; i++) {
        let train_samples_1  = samples.slice(0, i * fold_size);
        let validate_samples = samples.slice(i * fold_size, (i + 1) * fold_size);
        let train_samples_2  = samples.slice((i + 1) * fold_size, samples.length);
        let train_samples    = train_samples_1.concat(train_samples_2);

        let tree: TreeNode = ID3(train_samples, attributes);
        let rules: Rule[] = tree.get_rules();
        
        let fold_correct_class: number = 0;
        if (tree != null) {
            validate_samples.forEach((sample) => {
                total_tested++;
                let appropriate_rule = rules.find((rule) => 
                    rule.pairs.every((pair) =>
                        sample.get_value(pair.name) == pair.value
                    )
                );
                if (!appropriate_rule) {
                    console.warn("No rule was found.");
                    return;
                }

                // if (tree.classify(sample) == sample.result) {
                if (appropriate_rule.result == sample.result) {
                    fold_correct_class++;
                }
            })
        }
        
        // console.log("Fold #" + i + ": " + (100 * fold_correct_class / validate_samples.length) + "%")
        correct_class += fold_correct_class;
    }

    console.log("Prediction rate: " + (100 * correct_class / total_tested) + "%")
}