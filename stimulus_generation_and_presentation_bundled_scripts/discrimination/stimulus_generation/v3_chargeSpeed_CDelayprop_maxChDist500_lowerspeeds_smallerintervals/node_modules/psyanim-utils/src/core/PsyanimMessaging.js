export default class PsyanimMessaging {

    static getHeader(buffer) {

        let data = new Uint32Array(buffer);

        return data[0];
    }

    static setHeader(buffer, value) {

        let data = new Uint32Array(buffer);

        data[0] = value;
    }
}

PsyanimMessaging.MESSAGE_HEADER_LENGTH = 4; // bytes

PsyanimMessaging.MESSAGE_TYPES = {
    ANIMATION_CLIP: 0x0001, 
};