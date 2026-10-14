import type { Fluid } from "./fluid";

function sigmoid(x: number){
    return 1 / (1 + Math.exp(-x));
}

function rb_gradient_hex(x: number){
    let f = sigmoid(x) * 2 - 1;

    if(f > 0){
        return `#0000FF${Math.floor(f*256).toString(16).padStart(2, "0")}`;
    }else{
        return `#FF0000${Math.floor(-f*256).toString(16).padStart(2, "0")}`;
    }
}

export class Visualizer {
    fluid: Fluid

    constructor(fluid: Fluid){
        this.fluid = fluid;
    }

    draw_grid(ctx: CanvasRenderingContext2D){
        let cx = (a: number) => a * ctx.canvas.width;
        let cy = (a: number) => a * ctx.canvas.height;

        for(let y = 0; y < this.fluid.height; y++){
            for(let x = 0; x < this.fluid.width; x++){
                ctx.lineWidth = 0.5;
                ctx.beginPath();
                ctx.moveTo(cx(x / this.fluid.width), cy(y / this.fluid.height));
                ctx.lineTo(cx(x / this.fluid.width), cy((y + 1) / this.fluid.height))
                ctx.moveTo(cx(x / this.fluid.width), cy(y / this.fluid.height));
                ctx.lineTo(cx((x + 1) / this.fluid.width), cy(y / this.fluid.height))
                ctx.stroke();
            }
        }

        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(cx(1), cy(0));
        ctx.lineTo(cx(1), cy(1))
        ctx.moveTo(cx(0), cy(1));
        ctx.lineTo(cx(1), cy(1));
        ctx.stroke();
    }

    draw_velocities(ctx: CanvasRenderingContext2D){
        let cx = (a: number) => a * ctx.canvas.width;
        let cy = (a: number) => a * ctx.canvas.height;

        ctx.lineWidth = 1;

        for(let y = 0; y < this.fluid.height; y++){
            for(let x = 0; x < this.fluid.width + 1; x++){
                ctx.beginPath();
                ctx.moveTo(cx(x / this.fluid.width), cy((y + 0.5) / this.fluid.height));
                ctx.lineTo(cx((x + this.fluid.vx[y][x] / 2) / this.fluid.width), cy((y + 0.5) / this.fluid.height));
                ctx.stroke();
            }
        }
        for(let y = 0; y < this.fluid.height + 1; y++){
            for(let x = 0; x < this.fluid.width; x++){
                ctx.beginPath();
                ctx.moveTo(cx((x + 0.5) / this.fluid.width), cy(y / this.fluid.height));
                ctx.lineTo(cx((x + 0.5) / this.fluid.width), cy((y + this.fluid.vy[y][x] / 2) / this.fluid.height));
                ctx.stroke();
            }
        }
    }

    draw_divergence(ctx: CanvasRenderingContext2D){
        let cx = (a: number) => a * ctx.canvas.width;
        let cy = (a: number) => a * ctx.canvas.height;

        let divergence = this.fluid.compute_divergence();

        for(let y = 0; y < this.fluid.height; y++){
            for(let x = 0; x < this.fluid.width; x++){
                ctx.fillStyle = rb_gradient_hex(divergence[y][x] * 0.7);
                ctx.fillRect(cx(x / this.fluid.width), cy(y / this.fluid.height), cx(1 / this.fluid.width), cy(1 / this.fluid.height));
            }
        }
    }
}