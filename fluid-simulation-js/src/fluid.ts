export class Fluid {
    width: number
    height: number
    vx: number[][]
    vy: number[][]

    constructor(width: number, height: number){
        this.width = width;
        this.height = height;

        this.vx = [];
        this.vy = [];

        for(let y = 0; y < this.height; y++){
            let row: number[] = [];
            for(let x = 0; x < this.width + 1; x++){
                row.push(Math.random() * 2 - 1);
            }
            this.vx.push(row);
        }
        for(let y = 0; y < this.height + 1; y++){
            let row: number[] = [];
            for(let x = 0; x < this.width; x++){
                row.push(Math.random() * 2 - 1);
            }
            this.vy.push(row);
        }
    }

    compute_divergence(){
        let divergence: number[][] = [];
        for(let y = 0; y < this.height; y++){
            let row: number[] = [];
            for(let x = 0; x < this.width; x++){
                let div = (this.vx[y][x+1] - this.vx[y][x]) + (this.vy[y+1][x] - this.vy[y][x]);
                row.push(div);
            }
            divergence.push(row);
        }
        return divergence;
    }
}