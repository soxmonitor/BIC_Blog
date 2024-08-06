let order = 4;
let N = Math.pow(2, order);
let total = N * N;
let path = [];
let counter = 0;

function setup() {
    createCanvas(400, 400);
    colorMode(HSB, 360, 255, 255);
    background(0);
    let len = width / N;

    for (let i = 0; i < total; i++) {
        path[i] = hilbert(i);
        path[i].mult(len);
        path[i].add(len / 2, len / 2);
    }

    frameRate(30);
}

function draw() {
    background(0);
    stroke(255, 0, 255);
    strokeWeight(2);
    noFill();

    beginShape();
    for (let i = 0; i <= counter; i++) {
        vertex(path[i].x, path[i].y);
    }
    endShape();

    if (counter < total - 1) {
        counter++;
    } else {
        noLoop(); // Stop the loop when we reach the end of the path
    }
}

function hilbert(i) {
    let points = [
        createVector(0, 0),
        createVector(0, 1),
        createVector(1, 1),
        createVector(1, 0)
    ];

    let index = i & 3;
    let v = points[index].copy();

    for (let j = 1; j < order; j++) {
        i >>= 2;
        index = i & 3;
        let len = pow(2, j);
        let temp;

        switch (index) {
            case 0:
                temp = v.x;
                v.x = v.y;
                v.y = temp;
                break;
            case 1:
                v.y += len;
                break;
            case 2:
                v.x += len;
                v.y += len;
                break;
            case 3:
                temp = len - 1 - v.x;
                v.x = len - 1 - v.y;
                v.y = temp;
                v.x += len;
                break;
        }
    }
    return v;
}
